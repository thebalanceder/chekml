#include "CSS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

// Precomputed constants
static const double INV_A_CUBED = 1.0 / (A * A * A);

// Fast Xorshift random number generator state
static unsigned int xorshift_state = 1;

// Initialize Xorshift seed
static inline void init_xorshift(unsigned int seed) {
    xorshift_state = seed ? seed : 1;
}

// Fast Xorshift random number generator
static inline unsigned int xorshift() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
inline double rand_double_css(double min, double max) {
    return min + (max - min) * ((double)xorshift() / (double)0xFFFFFFFF);
}

// Fast inverse square root approximation with two Newton iterations
static inline double fast_inv_sqrt(double x) {
    union { double d; uint64_t i; } u = { x };
    u.i = 0x5fe6ec85e7de30da - (u.i >> 1);
    double y = u.d;
    y *= (1.5 - 0.5 * x * y * y); // First Newton iteration
    y *= (1.5 - 0.5 * x * y * y); // Second Newton iteration
    return y;
}

// Initialize charged particles with SoA layout
void initialize_charged_particles(Optimizer *opt) {
    register int i, j;
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double_css(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Calculate forces with SIMD and SoA
void calculate_forces(Optimizer *opt, double *forces) {
    static double *charges = NULL;
    static double *fitness_cache = NULL;
    static double *positions[32] = { NULL }; // SoA, max dim 32
    if (!charges) {
        charges = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
        fitness_cache = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
        for (int j = 0; j < opt->dim; j++) {
            positions[j] = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
        }
    }
    // Transpose positions to SoA
    register int i, j;
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            positions[j][i] = opt->population[i].position[j];
        }
        fitness_cache[i] = opt->population[i].fitness;
    }
    double fitworst = -INFINITY, fitbest = INFINITY;
    for (i = 0; i < opt->population_size; i++) {
        if (fitness_cache[i] > fitworst) fitworst = fitness_cache[i];
        if (fitness_cache[i] < fitbest) fitbest = fitness_cache[i];
    }
    // Compute charges
    double charge_scale = (fitbest == fitworst) ? 1.0 : 1.0 / (fitbest - fitworst);
    for (i = 0; i < opt->population_size; i++) {
        charges[i] = (fitbest == fitworst) ? 1.0 : (fitness_cache[i] - fitworst) * charge_scale;
    }
    // Initialize forces
    memset(forces, 0, opt->population_size * opt->dim * sizeof(double));
    // SIMD force calculation
    __m256d zero = _mm256_setzero_pd();
    __m256d a_cubed = _mm256_set1_pd(INV_A_CUBED);
    __m256d epsilon = _mm256_set1_pd(EPSILON);
    for (i = 0; i < opt->population_size; i++) {
        for (j = i + 1; j < opt->population_size; j++) {
            double r_ij = 0.0, r_ij_norm = 0.0;
            __m256d sum_r_ij = zero, sum_r_ij_norm = zero;
            for (int k = 0; k < opt->dim; k += 4) {
                if (k + 4 > opt->dim) break; // Handle non-aligned dims
                __m256d pos_i = _mm256_load_pd(positions[k] + i);
                __m256d pos_j = _mm256_load_pd(positions[k] + j);
                __m256d diff = _mm256_sub_pd(pos_i, pos_j);
                sum_r_ij = _mm256_fmadd_pd(diff, diff, sum_r_ij);
                __m256d sum_pos = _mm256_add_pd(pos_i, pos_j);
                __m256d mid = _mm256_mul_pd(sum_pos, _mm256_set1_pd(0.5));
                __m256d best = _mm256_load_pd(opt->best_solution.position + k);
                __m256d norm_diff = _mm256_sub_pd(mid, best);
                sum_r_ij_norm = _mm256_fmadd_pd(norm_diff, norm_diff, sum_r_ij_norm);
            }
            double r_ij_vec[4], r_ij_norm_vec[4];
            _mm256_store_pd(r_ij_vec, sum_r_ij);
            _mm256_store_pd(r_ij_norm_vec, sum_r_ij_norm);
            for (int k = 0; k < 4; k++) {
                r_ij += r_ij_vec[k];
                r_ij_norm += r_ij_norm_vec[k];
            }
            // Handle remaining dimensions scalarly
            for (int k = (opt->dim / 4) * 4; k < opt->dim; k++) {
                double diff = positions[k][i] - positions[k][j];
                r_ij += diff * diff;
                double norm_diff = (positions[k][i] + positions[k][j]) / 2.0 - opt->best_solution.position[k];
                r_ij_norm += norm_diff * norm_diff;
            }
            r_ij = 1.0 / fast_inv_sqrt(r_ij);
            r_ij_norm = r_ij / (1.0 / fast_inv_sqrt(r_ij_norm) + EPSILON);
            double p_ij = (fitness_cache[i] < fitness_cache[j] || rand_double_css(0.0, 1.0) < 0.5) ? 1.0 : 0.0;
            double force_term = (r_ij < A) ? (charges[i] * charges[j] * INV_A_CUBED) * r_ij : (charges[i] * charges[j] / (r_ij * r_ij));
            int i_base = i * opt->dim, j_base = j * opt->dim;
            for (int k = 0; k < opt->dim; k++) {
                double force = p_ij * force_term * (positions[k][i] - positions[k][j]);
                forces[i_base + k] += force;
                forces[j_base + k] -= force;
            }
        }
    }
}

// Update positions with SIMD and minimal branching
void css_update_positions(Optimizer *opt, double *forces) {
    static double *velocities = NULL;
    if (!velocities) {
        velocities = (double *)_mm_malloc(opt->population_size * opt->dim * sizeof(double), 32);
        memset(velocities, 0, opt->population_size * opt->dim * sizeof(double));
    }
    double dt = 1.0;
    __m256d kv = _mm256_set1_pd(KV);
    __m256d ka = _mm256_set1_pd(KA);
    __m256d dt_vec = _mm256_set1_pd(dt);
    register int i, j;
    for (i = 0; i < opt->population_size; i++) {
        double rand1 = rand_double_css(0.0, 1.0);
        double rand2 = rand_double_css(0.0, 1.0);
        __m256d r1 = _mm256_set1_pd(rand1);
        __m256d r2 = _mm256_set1_pd(rand2);
        int base_idx = i * opt->dim;
        for (j = 0; j < opt->dim; j += 4) {
            if (j + 4 > opt->dim) break; // Handle non-aligned dims
            __m256d vel = _mm256_load_pd(velocities + base_idx + j);
            __m256d force = _mm256_load_pd(forces + base_idx + j);
            vel = _mm256_fmadd_pd(r1, _mm256_mul_pd(kv, vel), _mm256_mul_pd(r2, _mm256_mul_pd(ka, force)));
            __m256d pos = _mm256_load_pd(opt->population[i].position + j);
            pos = _mm256_fmadd_pd(vel, dt_vec, pos);
            __m256d min_bound = _mm256_load_pd(opt->bounds + 2 * j);
            __m256d max_bound = _mm256_load_pd(opt->bounds + 2 * j + 1);
            pos = _mm256_max_pd(min_bound, _mm256_min_pd(max_bound, pos));
            _mm256_store_pd(opt->population[i].position + j, pos);
            _mm256_store_pd(velocities + base_idx + j, vel);
        }
        // Handle remaining dimensions scalarly
        for (j = (opt->dim / 4) * 4; j < opt->dim; j++) {
            int idx = base_idx + j;
            velocities[idx] = rand1 * KV * velocities[idx] + rand2 * KA * forces[idx];
            opt->population[i].position[j] += velocities[idx] * dt;
            opt->population[i].position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], opt->population[i].position[j]));
        }
    }
}

// Update charged memory with best solutions
void update_charged_memory(Optimizer *opt, double *forces) {
    int cm_size = (int)(CM_SIZE_RATIO * opt->population_size);
    static Solution *charged_memory = NULL;
    static int memory_size = 0;
    if (!charged_memory) {
        charged_memory = (Solution *)malloc(cm_size * sizeof(Solution));
        for (int i = 0; i < cm_size; i++) {
            charged_memory[i].position = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
        }
        memory_size = 0;
    }
    static int *indices = NULL;
    if (!indices) {
        indices = (int *)malloc(opt->population_size * sizeof(int));
    }
    register int i, j;
    for (i = 0; i < opt->population_size; i++) indices[i] = i;
    // Partial selection sort with unrolling
    for (i = 0; i < cm_size && i < opt->population_size; i++) {
        int min_idx = i;
        for (j = i + 1; j < opt->population_size; j += 4) {
            if (j < opt->population_size && opt->population[indices[j]].fitness < opt->population[indices[min_idx]].fitness) {
                min_idx = j;
            }
            if (j + 1 < opt->population_size && opt->population[indices[j + 1]].fitness < opt->population[indices[min_idx]].fitness) {
                min_idx = j + 1;
            }
            if (j + 2 < opt->population_size && opt->population[indices[j + 2]].fitness < opt->population[indices[min_idx]].fitness) {
                min_idx = j + 2;
            }
            if (j + 3 < opt->population_size && opt->population[indices[j + 3]].fitness < opt->population[indices[min_idx]].fitness) {
                min_idx = j + 3;
            }
        }
        int temp = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp;
    }
    // Update memory with SIMD copying
    for (i = 0; i < cm_size && i < opt->population_size; i++) {
        if (memory_size < cm_size) {
            for (j = 0; j < opt->dim; j += 4) {
                if (j + 4 > opt->dim) break;
                __m256d src = _mm256_load_pd(opt->population[indices[i]].position + j);
                _mm256_store_pd(charged_memory[memory_size].position + j, src);
            }
            for (j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                charged_memory[memory_size].position[j] = opt->population[indices[i]].position[j];
            }
            charged_memory[memory_size].fitness = opt->population[indices[i]].fitness;
            memory_size++;
        } else {
            int worst_idx = 0;
            double worst_fitness = charged_memory[0].fitness;
            for (j = 1; j < memory_size; j++) {
                if (charged_memory[j].fitness > worst_fitness) {
                    worst_fitness = charged_memory[j].fitness;
                    worst_idx = j;
                }
            }
            if (opt->population[indices[i]].fitness < worst_fitness) {
                for (j = 0; j < opt->dim; j += 4) {
                    if (j + 4 > opt->dim) break;
                    __m256d src = _mm256_load_pd(opt->population[indices[i]].position + j);
                    _mm256_store_pd(charged_memory[worst_idx].position + j, src);
                }
                for (j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                    charged_memory[worst_idx].position[j] = opt->population[indices[i]].position[j];
                }
                charged_memory[worst_idx].fitness = opt->population[indices[i]].fitness;
            }
        }
    }
}

// Main optimization function
void CSS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    init_xorshift((unsigned int)time(NULL));
    initialize_charged_particles(opt);
    double *forces = (double *)_mm_malloc(opt->population_size * opt->dim * sizeof(double), 32);
    register int iter, i, j;
    for (iter = 0; iter < opt->max_iter; iter++) {
        for (i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (j = 0; j < opt->dim; j += 4) {
                    if (j + 4 > opt->dim) break;
                    __m256d src = _mm256_load_pd(opt->population[i].position + j);
                    _mm256_store_pd(opt->best_solution.position + j, src);
                }
                for (j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        calculate_forces(opt, forces);
        css_update_positions(opt, forces);
        update_charged_memory(opt, forces);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    _mm_free(forces);
}
