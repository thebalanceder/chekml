#include "JOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <stdint.h> // Added for uint64_t
#include <immintrin.h> // AVX2
#include <omp.h>

// Xorshift128+ random number generator
typedef struct {
    uint64_t s[2];
} Xorshift128Plus;

static inline uint64_t xorshift128plus_next(Xorshift128Plus *state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0 ^ (s0 >> 26);
    state->s[1] = s1;
    return s0 + s1;
}

static inline double xorshift128plus_double(Xorshift128Plus *state, double min, double max) {
    const double norm = 1.0 / (1ULL << 52);
    return min + (max - min) * ((double)(xorshift128plus_next(state) >> 12) * norm);
}

// rand_double_joa for compatibility
double rand_double_joa(double min, double max) {
    static Xorshift128Plus rng = { .s = {0, 0} };
    if (rng.s[0] == 0) {
        rng.s[0] = (uint64_t)time(NULL) ^ 0xDEADBEEF;
        rng.s[1] = (uint64_t)time(NULL) ^ 0xCAFEBABE;
    }
    return xorshift128plus_double(&rng, min, max);
}

// Initialize subpopulations
void initialize_subpopulations(Optimizer *opt) {
    Xorshift128Plus rng = { .s = {(uint64_t)time(NULL) ^ 0xDEADBEEF, (uint64_t)time(NULL) ^ 0xCAFEBABE} };
    const int total_pop_size = NUM_SUBPOPULATIONS * POPULATION_SIZE_PER_SUBPOP;
    
    if (opt->population_size != total_pop_size) {
        fprintf(stderr, "Population size mismatch: expected %d, got %d\n", total_pop_size, opt->population_size);
        exit(1);
    }

    const int dim = opt->dim;
    double *bounds = opt->bounds;
    Solution *pop = opt->population;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < total_pop_size; i++) {
        double *pos = pop[i].position;
        _mm_prefetch((char *)(pos + dim), _MM_HINT_T0); // Prefetch next position
        for (int k = 0; k < dim; k++) {
            pos[k] = xorshift128plus_double(&rng, bounds[2 * k], bounds[2 * k + 1]);
        }
        pop[i].fitness = INFINITY;
    }

    opt->optimize = JOA_optimize;
}

// Evaluate fitness with SIMD
void evaluate_subpopulations(Optimizer *opt, double (*objective_function)(double *)) {
    const int dim = opt->dim;
    Solution *pop = opt->population;
    double best_fitness = opt->best_solution.fitness;
    double *best_pos = opt->best_solution.position;

    #pragma omp parallel for schedule(dynamic, 1) reduction(min:best_fitness)
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;
            double *pos = pop[idx].position;
            _mm_prefetch((char *)(pop[idx + 1].position), _MM_HINT_T0); // Prefetch next
            double fitness = objective_function(pos);
            pop[idx].fitness = fitness;

            if (fitness < best_fitness) {
                best_fitness = fitness;
                #pragma omp critical
                {
                    if (fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = fitness;
                        // SIMD copy for best_pos
                        for (int k = 0; k < dim; k += 4) {
                            if (k + 4 <= dim) {
                                __m256d vec = _mm256_loadu_pd(pos + k);
                                _mm256_storeu_pd(best_pos + k, vec);
                            } else {
                                for (; k < dim; k++) {
                                    best_pos[k] = pos[k];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Update subpopulations with SIMD
void update_subpopulations(Optimizer *opt, double *temp_direction) {
    Xorshift128Plus rng = { .s = {(uint64_t)time(NULL) ^ 0xDEADBEEF, (uint64_t)time(NULL) ^ 0xCAFEBABE} };
    const int dim = opt->dim;
    Solution *pop = opt->population;
    double *bounds = opt->bounds;
    const __m256d mutation_vec = _mm256_set1_pd(MUTATION_RATE);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        int other_subpop_idx = (int)(xorshift128plus_next(&rng) % NUM_SUBPOPULATIONS);
        if (other_subpop_idx == i) {
            other_subpop_idx = (other_subpop_idx + 1) % NUM_SUBPOPULATIONS;
        }
        int other_start_idx = other_subpop_idx * POPULATION_SIZE_PER_SUBPOP;

        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;
            int other_ind_idx = other_start_idx + ((int)(xorshift128plus_next(&rng) % POPULATION_SIZE_PER_SUBPOP));
            double *pos = pop[idx].position;
            double *other_pos = pop[other_ind_idx].position;
            _mm_prefetch((char *)(pop[idx + 1].position), _MM_HINT_T0); // Prefetch next

            // SIMD update
            for (int k = 0; k < dim; k += 4) {
                if (k + 4 <= dim) {
                    __m256d pos_vec = _mm256_loadu_pd(pos + k);
                    __m256d other_vec = _mm256_loadu_pd(other_pos + k);
                    __m256d diff = _mm256_sub_pd(other_vec, pos_vec);
                    __m256d update = _mm256_mul_pd(diff, mutation_vec);
                    pos_vec = _mm256_add_pd(pos_vec, update);
                    _mm256_storeu_pd(pos + k, pos_vec);
                } else {
                    for (; k < dim; k++) {
                        temp_direction[k] = other_pos[k] - pos[k];
                        pos[k] += MUTATION_RATE * temp_direction[k];
                    }
                }
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Main optimization function
void JOA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    const int dim = opt->dim;

    // Allocate temporary direction array
    double *temp_direction = (double *)aligned_alloc(32, (dim + 3) & ~3); // Pad for SIMD
    if (!temp_direction) {
        fprintf(stderr, "Memory allocation failed for temp_direction\n");
        exit(1);
    }

    initialize_subpopulations(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_subpopulations(opt, objective_function);
        update_subpopulations(opt, temp_direction);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free only JOA-specific memory
    free(temp_direction);
}
