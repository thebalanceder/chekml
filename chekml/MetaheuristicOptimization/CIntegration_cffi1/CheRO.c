#include "CheRO.h"
#include <stdlib.h>
#include <time.h>

// Initialize Xorshift128+ RNG
void cro_init_rng(CROOptimizer *cro) {
    cro->rng_state[0] = (unsigned long long)time(NULL) ^ 0xDEADBEEFDEADBEEF;
    cro->rng_state[1] = (unsigned long long)time(NULL) ^ 0xCAFEBABECAFEBABE;
    cro->rng_index = CRO_RNG_BUFFER_SIZE; // Force initial fill
}

// Xorshift128+ RNG for high-quality random numbers
static inline unsigned long long xorshift128p(CROOptimizer *cro) {
    unsigned long long x = cro->rng_state[0];
    unsigned long long const y = cro->rng_state[1];
    cro->rng_state[0] = y;
    x ^= x << 23;
    cro->rng_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return cro->rng_state[1] + y;
}

// Fill RNG buffer with random doubles in [0, 1]
void cro_fill_rng_buffer(CROOptimizer *cro) {
    for (int i = 0; i < CRO_RNG_BUFFER_SIZE; i++) {
        cro->rng_buffer[i] = (double)xorshift128p(cro) / (double)0xFFFFFFFFFFFFFFFF;
    }
    cro->rng_index = 0;
}

// Get random double from buffer, refilling if needed
static inline double cro_rand_double(CROOptimizer *cro, double min, double max) {
    if (cro->rng_index >= CRO_RNG_BUFFER_SIZE) {
        cro_fill_rng_buffer(cro);
    }
    return min + (max - min) * cro->rng_buffer[cro->rng_index++];
}

// Initialize molecules with random positions and kinetic energies
void initialize_molecules(CROOptimizer *cro) {
    cro->positions = (double *)_mm_malloc(CRO_MAX_POPULATION * cro->opt->dim * sizeof(double), CRO_ALIGN);
    cro->pes = (double *)_mm_malloc(CRO_MAX_POPULATION * sizeof(double), CRO_ALIGN);
    cro->kes = (double *)_mm_malloc(CRO_MAX_POPULATION * sizeof(double), CRO_ALIGN);
    cro->temp_solution1 = (double *)_mm_malloc(cro->opt->dim * sizeof(double), CRO_ALIGN);
    cro->temp_solution2 = (double *)_mm_malloc(cro->opt->dim * sizeof(double), CRO_ALIGN);
    cro->rng_buffer = (double *)_mm_malloc(CRO_RNG_BUFFER_SIZE * sizeof(double), CRO_ALIGN);
    
    for (int i = 0; i < cro->population_size; i++) {
        double *pos = cro->positions + i * cro->opt->dim;
        for (int j = 0; j < cro->opt->dim; j++) {
            pos[j] = cro_rand_double(cro, cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->pes[i] = cro->objective_function(pos);
        cro->kes[i] = CRO_INITIAL_KE;
        if (cro->pes[i] < cro->opt->best_solution.fitness) {
            cro->opt->best_solution.fitness = cro->pes[i];
            for (int j = 0; j < cro->opt->dim; j++) {
                cro->opt->best_solution.position[j] = pos[j];
            }
        }
    }
    cro->buffer = CRO_BUFFER_INITIAL;
}

// Evaluate potential energy (fitness) for modified molecules
void evaluate_molecules(CROOptimizer *cro) {
    for (int i = 0; i < cro->population_size; i++) {
        double *pos = cro->positions + i * cro->opt->dim;
        double pe = cro->objective_function(pos);
        cro->pes[i] = pe;
        if (pe < cro->opt->best_solution.fitness) {
            cro->opt->best_solution.fitness = pe;
            for (int j = 0; j < cro->opt->dim; j++) {
                cro->opt->best_solution.position[j] = pos[j];
            }
        }
    }
}

// SIMD-optimized on-wall collision (local search)
void on_wall_collision(CROOptimizer *cro, int index) {
    double *pos = cro->positions + index * cro->opt->dim;
    double *new_solution = cro->temp_solution1;
    double r = cro_rand_double(cro, 0.0, 1.0);
    int dim = cro->opt->dim;
    
    #pragma omp simd aligned(pos, new_solution : CRO_ALIGN)
    for (int j = 0; j < dim; j++) {
        double delta = r * (cro->opt->bounds[2 * j + 1] - cro->opt->bounds[2 * j]) * cro_rand_double(cro, -1.0, 1.0);
        new_solution[j] = pos[j] + delta;
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe = cro->pes[index];
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe + cro->kes[index] >= new_pe) {
        for (int j = 0; j < dim; j++) {
            pos[j] = new_solution[j];
        }
        cro->pes[index] = new_pe;
        cro->kes[index] = old_pe + cro->kes[index] - new_pe;
    }
}

// SIMD-optimized decomposition reaction (global exploration)
void decomposition(CROOptimizer *cro, int index) {
    if (cro->population_size >= CRO_MAX_POPULATION) return;
    
    double old_pe = cro->pes[index];
    if (cro->kes[index] + old_pe < CRO_ALPHA) return;
    
    double *pos = cro->positions + index * cro->opt->dim;
    double *split1 = cro->temp_solution1;
    double *split2 = cro->temp_solution2;
    int dim = cro->opt->dim;
    
    #pragma omp simd aligned(pos, split1, split2 : CRO_ALIGN)
    for (int j = 0; j < dim; j++) {
        double r = cro_rand_double(cro, -1.0, 1.0);
        split1[j] = pos[j] + CRO_SPLIT_RATIO * r;
        split2[j] = pos[j] - CRO_SPLIT_RATIO * r;
        split1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split1[j]));
        split2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split2[j]));
    }
    
    double pe1 = cro->objective_function(split1);
    double pe2 = cro->objective_function(split2);
    
    if (old_pe + cro->kes[index] >= pe1 + pe2) {
        for (int j = 0; j < dim; j++) {
            pos[j] = split1[j];
        }
        cro->pes[index] = pe1;
        cro->kes[index] = (old_pe + cro->kes[index] - pe1 - pe2) / 2.0;
        
        double *new_pos = cro->positions + cro->population_size * dim;
        for (int j = 0; j < dim; j++) {
            new_pos[j] = split2[j];
        }
        cro->pes[cro->population_size] = pe2;
        cro->kes[cro->population_size] = (old_pe + cro->kes[index] - pe1 - pe2) / 2.0;
        cro->population_size++;
    }
}

// SIMD-optimized inter-molecular collision (local search)
void inter_molecular_collision(CROOptimizer *cro, int index1, int index2) {
    double *pos1 = cro->positions + index1 * cro->opt->dim;
    double *pos2 = cro->positions + index2 * cro->opt->dim;
    double *new_solution1 = cro->temp_solution1;
    double *new_solution2 = cro->temp_solution2;
    double r1 = cro_rand_double(cro, 0.0, 1.0);
    double r2 = cro_rand_double(cro, 0.0, 1.0);
    int dim = cro->opt->dim;
    
    #pragma omp simd aligned(pos1, pos2, new_solution1, new_solution2 : CRO_ALIGN)
    for (int j = 0; j < dim; j++) {
        new_solution1[j] = pos1[j] + r1 * cro_rand_double(cro, -1.0, 1.0);
        new_solution2[j] = pos2[j] + r2 * cro_rand_double(cro, -1.0, 1.0);
        new_solution1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution1[j]));
        new_solution2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution2[j]));
    }
    
    double old_pe1 = cro->pes[index1];
    double old_pe2 = cro->pes[index2];
    double new_pe1 = cro->objective_function(new_solution1);
    double new_pe2 = cro->objective_function(new_solution2);
    
    if (old_pe1 + old_pe2 + cro->kes[index1] + cro->kes[index2] >= new_pe1 + new_pe2) {
        for (int j = 0; j < dim; j++) {
            pos1[j] = new_solution1[j];
            pos2[j] = new_solution2[j];
        }
        cro->pes[index1] = new_pe1;
        cro->pes[index2] = new_pe2;
        double total_ke = old_pe1 + old_pe2 + cro->kes[index1] + cro->kes[index2] - new_pe1 - new_pe2;
        cro->kes[index1] = total_ke * cro_rand_double(cro, 0.0, 1.0);
        cro->kes[index2] = total_ke - cro->kes[index1];
    }
}

// SIMD-optimized synthesis reaction (global exploration)
void synthesis(CROOptimizer *cro, int index1, int index2) {
    double *pos1 = cro->positions + index1 * cro->opt->dim;
    double *pos2 = cro->positions + index2 * cro->opt->dim;
    double *new_solution = cro->temp_solution1;
    int dim = cro->opt->dim;
    
    #pragma omp simd aligned(pos1, pos2, new_solution : CRO_ALIGN)
    for (int j = 0; j < dim; j++) {
        new_solution[j] = (pos1[j] + pos2[j]) * 0.5;
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe1 = cro->pes[index1];
    double old_pe2 = cro->pes[index2];
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe1 + old_pe2 + cro->kes[index1] + cro->kes[index2] >= new_pe + CRO_BETA) {
        for (int j = 0; j < dim; j++) {
            pos1[j] = new_solution[j];
        }
        cro->pes[index1] = new_pe;
        cro->kes[index1] = old_pe1 + old_pe2 + cro->kes[index1] + cro->kes[index2] - new_pe;
        cro->buffer += cro_rand_double(cro, 0.0, 1.0) * cro->kes[index1];
        
        // Shift molecules to remove index2
        for (int i = index2; i < cro->population_size - 1; i++) {
            double *dest = cro->positions + i * dim;
            for (int j = 0; j < dim; j++) {
                dest[j] = cro->positions[(i + 1) * dim + j];
            }
            cro->pes[i] = cro->pes[i + 1];
            cro->kes[i] = cro->kes[i + 1];
        }
        cro->population_size--;
    }
}

// Quickselect to find worst molecules
static void quickselect(double *pes, int *indices, int left, int right, int k) {
    if (left >= right) return;
    
    int pivot_idx = left + (right - left) / 2;
    double pivot = pes[indices[pivot_idx]];
    int i = left, j = right;
    
    while (i <= j) {
        while (pes[indices[i]] < pivot) i++;
        while (pes[indices[j]] > pivot) j--;
        if (i <= j) {
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
            i++;
            j--;
        }
    }
    
    if (k <= j) quickselect(pes, indices, left, j, k);
    else if (k >= i) quickselect(pes, indices, i, right, k);
}

// Elimination phase (replace worst molecules)
void elimination_phase_chero(CROOptimizer *cro) {
    int worst_count = (int)(CRO_ELIMINATION_RATIO * cro->population_size);
    if (worst_count == 0) return;
    
    int *indices = (int *)malloc(cro->population_size * sizeof(int));
    for (int i = 0; i < cro->population_size; i++) {
        indices[i] = i;
    }
    
    // Find the worst_count highest pes
    quickselect(cro->pes, indices, 0, cro->population_size - 1, cro->population_size - worst_count);
    
    // Replace worst molecules
    int dim = cro->opt->dim;
    for (int i = 0; i < worst_count; i++) {
        int idx = indices[cro->population_size - 1 - i];
        double *pos = cro->positions + idx * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] = cro_rand_double(cro, cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->pes[idx] = cro->objective_function(pos);
        cro->kes[idx] = CRO_INITIAL_KE;
        cro->buffer += cro_rand_double(cro, 0.0, 1.0) * cro->kes[idx];
    }
    
    free(indices);
}

// Main optimization function
void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    CROOptimizer cro;
    cro.opt = opt;
    cro.population_size = opt->population_size;
    cro.objective_function = objective_function;
    cro_init_rng(&cro);
    
    initialize_molecules(&cro);
    
    double prev_best = cro.opt->best_solution.fitness;
    int stall_count = 0;
    const int max_stalls = 10; // Early stopping after 10 unchanged iterations
    
    for (int iter = 0; iter < cro.opt->max_iter; iter++) {
        evaluate_molecules(&cro);
        
        int i = 0;
        while (i < cro.population_size) {
            if (cro_rand_double(&cro, 0.0, 1.0) < CRO_MOLE_COLL && i < cro.population_size - 1) {
                int index1 = i;
                int index2 = i + 1;
                if (cro_rand_double(&cro, 0.0, 1.0) < 0.5) {
                    inter_molecular_collision(&cro, index1, index2);
                } else if (index1 < cro.population_size && index2 < cro.population_size) {
                    synthesis(&cro, index1, index2);
                }
                i += 2;
            } else {
                if (i < cro.population_size) {
                    if (cro_rand_double(&cro, 0.0, 1.0) < 0.5) {
                        on_wall_collision(&cro, i);
                    } else {
                        decomposition(&cro, i);
                    }
                }
                i++;
            }
        }
        
        elimination_phase_chero(&cro);
        printf("Iteration %d: Best Value = %f\n", iter + 1, cro.opt->best_solution.fitness);
        
        // Early stopping check
        if (fabs(cro.opt->best_solution.fitness - prev_best) < 1e-6) {
            stall_count++;
            if (stall_count >= max_stalls) break;
        } else {
            stall_count = 0;
            prev_best = cro.opt->best_solution.fitness;
        }
    }
    
    // Cleanup
    _mm_free(cro.positions);
    _mm_free(cro.pes);
    _mm_free(cro.kes);
    _mm_free(cro.temp_solution1);
    _mm_free(cro.temp_solution2);
    _mm_free(cro.rng_buffer);
}
