#include "CheRO.h"
#include <stdlib.h>
#include <time.h>

// Initialize Xorshift RNG
void cro_init_rng(CROOptimizer *cro) {
    cro->rng_state = (unsigned int)time(NULL) ^ 0xDEADBEEF;
}

// Xorshift RNG for fast random number generation
static inline unsigned int xorshift(CROOptimizer *cro) {
    unsigned int x = cro->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    cro->rng_state = x;
    return x;
}

// Generate random double between min and max
static inline double cro_rand_double(CROOptimizer *cro, double min, double max) {
    return min + (max - min) * (xorshift(cro) / (double)0xFFFFFFFF);
}

// Initialize molecules with random positions and kinetic energies
void initialize_molecules(CROOptimizer *cro) {
    cro->molecules = (Molecule *)malloc(CRO_MAX_POPULATION * sizeof(Molecule));
    cro->temp_solution1 = (double *)malloc(cro->opt->dim * sizeof(double));
    cro->temp_solution2 = (double *)malloc(cro->opt->dim * sizeof(double));
    for (int i = 0; i < cro->population_size; i++) {
        cro->molecules[i].position = (double *)malloc(cro->opt->dim * sizeof(double));
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[i].position[j] = cro_rand_double(cro, cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->molecules[i].ke = CRO_INITIAL_KE;
        cro->molecules[i].pe = cro->objective_function(cro->molecules[i].position);
    }
    cro->buffer = CRO_BUFFER_INITIAL;
}

// Evaluate potential energy (fitness) for all molecules
void evaluate_molecules(CROOptimizer *cro) {
    for (int i = 0; i < cro->population_size; i++) {
        double pe = cro->objective_function(cro->molecules[i].position);
        cro->molecules[i].pe = pe;
        if (pe < cro->opt->best_solution.fitness) {
            cro->opt->best_solution.fitness = pe;
            for (int j = 0; j < cro->opt->dim; j++) {
                cro->opt->best_solution.position[j] = cro->molecules[i].position[j];
            }
        }
    }
}

// On-wall ineffective collision (local search)
void on_wall_collision(CROOptimizer *cro, int index) {
    double *new_solution = cro->temp_solution1;
    double r = cro_rand_double(cro, 0.0, 1.0);
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution[j] = cro->molecules[index].position[j] + 
                          r * (cro->opt->bounds[2 * j + 1] - cro->opt->bounds[2 * j]) * cro_rand_double(cro, -1.0, 1.0);
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe = cro->molecules[index].pe;
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe + cro->molecules[index].ke >= new_pe) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index].position[j] = new_solution[j];
        }
        cro->molecules[index].pe = new_pe;
        cro->molecules[index].ke = old_pe + cro->molecules[index].ke - new_pe;
    }
}

// Decomposition reaction (global exploration)
void decomposition(CROOptimizer *cro, int index) {
    if (cro->population_size >= CRO_MAX_POPULATION) return;
    
    double old_pe = cro->molecules[index].pe;
    if (cro->molecules[index].ke + old_pe < CRO_ALPHA) return;
    
    double *split1 = cro->temp_solution1;
    double *split2 = cro->temp_solution2;
    for (int j = 0; j < cro->opt->dim; j++) {
        split1[j] = cro->molecules[index].position[j] + CRO_SPLIT_RATIO * cro_rand_double(cro, -1.0, 1.0);
        split2[j] = cro->molecules[index].position[j] - CRO_SPLIT_RATIO * cro_rand_double(cro, -1.0, 1.0);
        split1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split1[j]));
        split2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split2[j]));
    }
    
    double pe1 = cro->objective_function(split1);
    double pe2 = cro->objective_function(split2);
    
    if (old_pe + cro->molecules[index].ke >= pe1 + pe2) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index].position[j] = split1[j];
        }
        cro->molecules[index].pe = pe1;
        cro->molecules[index].ke = (old_pe + cro->molecules[index].ke - pe1 - pe2) / 2.0;
        
        cro->molecules[cro->population_size].position = (double *)malloc(cro->opt->dim * sizeof(double));
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[cro->population_size].position[j] = split2[j];
        }
        cro->molecules[cro->population_size].pe = pe2;
        cro->molecules[cro->population_size].ke = (old_pe + cro->molecules[index].ke - pe1 - pe2) / 2.0;
        cro->population_size++;
    }
}

// Inter-molecular ineffective collision (local search)
void inter_molecular_collision(CROOptimizer *cro, int index1, int index2) {
    double *new_solution1 = cro->temp_solution1;
    double *new_solution2 = cro->temp_solution2;
    double r1 = cro_rand_double(cro, 0.0, 1.0);
    double r2 = cro_rand_double(cro, 0.0, 1.0);
    
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution1[j] = cro->molecules[index1].position[j] + r1 * cro_rand_double(cro, -1.0, 1.0);
        new_solution2[j] = cro->molecules[index2].position[j] + r2 * cro_rand_double(cro, -1.0, 1.0);
        new_solution1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution1[j]));
        new_solution2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution2[j]));
    }
    
    double old_pe1 = cro->molecules[index1].pe;
    double old_pe2 = cro->molecules[index2].pe;
    double new_pe1 = cro->objective_function(new_solution1);
    double new_pe2 = cro->objective_function(new_solution2);
    
    if (old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke >= new_pe1 + new_pe2) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index1].position[j] = new_solution1[j];
            cro->molecules[index2].position[j] = new_solution2[j];
        }
        cro->molecules[index1].pe = new_pe1;
        cro->molecules[index2].pe = new_pe2;
        double total_ke = old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke - new_pe1 - new_pe2;
        cro->molecules[index1].ke = total_ke * cro_rand_double(cro, 0.0, 1.0);
        cro->molecules[index2].ke = total_ke - cro->molecules[index1].ke;
    }
}

// Synthesis reaction (global exploration)
void synthesis(CROOptimizer *cro, int index1, int index2) {
    double *new_solution = cro->temp_solution1;
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution[j] = (cro->molecules[index1].position[j] + cro->molecules[index2].position[j]) / 2.0;
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe1 = cro->molecules[index1].pe;
    double old_pe2 = cro->molecules[index2].pe;
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke >= new_pe + CRO_BETA) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index1].position[j] = new_solution[j];
        }
        cro->molecules[index1].pe = new_pe;
        cro->molecules[index1].ke = old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke - new_pe;
        cro->buffer += cro_rand_double(cro, 0.0, 1.0) * cro->molecules[index1].ke;
        
        free(cro->molecules[index2].position);
        for (int i = index2; i < cro->population_size - 1; i++) {
            cro->molecules[i] = cro->molecules[i + 1];
        }
        cro->population_size--;
    }
}

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    double fa = ((Molecule *)a)->pe;
    double fb = ((Molecule *)b)->pe;
    return (fa > fb) - (fa < fb);
}

// Elimination phase (replace worst molecules)
void elimination_phase_chero(CROOptimizer *cro) {
    int worst_count = (int)(CRO_ELIMINATION_RATIO * cro->population_size);
    if (worst_count == 0) return;
    
    // Sort molecules by fitness (descending)
    qsort(cro->molecules, cro->population_size, sizeof(Molecule), compare_fitness);
    
    // Replace worst molecules
    for (int i = 0; i < worst_count; i++) {
        int idx = cro->population_size - 1 - i;
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[idx].position[j] = cro_rand_double(cro, cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->molecules[idx].ke = CRO_INITIAL_KE;
        cro->molecules[idx].pe = cro->objective_function(cro->molecules[idx].position);
        cro->buffer += cro_rand_double(cro, 0.0, 1.0) * cro->molecules[idx].ke;
    }
}

// Main optimization function
void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    CROOptimizer cro;
    cro.opt = opt;
    cro.population_size = opt->population_size;
    cro.objective_function = objective_function;
    cro_init_rng(&cro);
    
    initialize_molecules(&cro);
    
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
    }
    
    // Cleanup
    for (int i = 0; i < cro.population_size; i++) {
        free(cro.molecules[i].position);
    }
    free(cro.molecules);
    free(cro.temp_solution1);
    free(cro.temp_solution2);
}
