#include "CRO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifndef NO_OPENMP
#include <omp.h>
#endif

// Thread-local Xorshift RNG state
static uint64_t rng_state;
#ifndef NO_OPENMP
#pragma omp threadprivate(rng_state)
#endif

// Initialize thread-local RNG
static void init_rng(unsigned int seed) {
#ifdef NO_OPENMP
    rng_state = seed ? seed : (unsigned int)time(NULL);
#else
    rng_state = seed ^ omp_get_thread_num();
    if (rng_state == 0) rng_state = (unsigned int)time(NULL) ^ omp_get_thread_num();
#endif
    if (rng_state == 0) rng_state = 1;  // Avoid zero state
}

// Fast Xorshift RNG (inlined macro)
#define XORSHIFT64() ({ \
    rng_state ^= rng_state >> 12; \
    rng_state ^= rng_state << 25; \
    rng_state ^= rng_state >> 27; \
    rng_state * 0x2545F4914F6CDD1DULL; \
})

// Macro for random double in [min, max]
#define RAND_DOUBLE(min, max) ((min) + ((max) - (min)) * ((double)XORSHIFT64() / (double)0xFFFFFFFFFFFFFFFFULL))

// Generate bulk random doubles
static void generate_random_doubles(double *buffer, int count, double min, double max) {
#ifdef __AVX2__
    __m256d min_vec = _mm256_set1_pd(min);
    __m256d range_vec = _mm256_set1_pd(max - min);
    __m256d divisor = _mm256_set1_pd((double)0xFFFFFFFFFFFFFFFFULL);
    for (int i = 0; i < count - 3; i += 4) {
        __m256d rand = _mm256_set_pd(
            (double)XORSHIFT64(), (double)XORSHIFT64(),
            (double)XORSHIFT64(), (double)XORSHIFT64()
        );
        __m256d result = _mm256_mul_pd(_mm256_div_pd(rand, divisor), range_vec);
        result = _mm256_add_pd(result, min_vec);
        _mm256_storeu_pd(buffer + i, result);
    }
    for (int i = (count / 4) * 4; i < count; i++) {
        buffer[i] = RAND_DOUBLE(min, max);
    }
#else
    for (int i = 0; i < count; i++) {
        buffer[i] = RAND_DOUBLE(min, max);
    }
#endif
}

// Initialize reefs randomly within bounds
void initialize_reefs(Optimizer *opt) {
    if (!opt || !opt->bounds || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer, bounds, or population\n");
        return;
    }

    int solutions_per_reef = opt->population_size / NUM_REEFS;
    if (opt->population_size % NUM_REEFS != 0 || solutions_per_reef <= 0) {
        fprintf(stderr, "Population size %d must be divisible by NUM_REEFS %d\n", opt->population_size, NUM_REEFS);
        return;
    }
    if (solutions_per_reef != POPULATION_SIZE) {
        fprintf(stderr, "Warning: Using %d solutions per reef instead of POPULATION_SIZE %d\n", solutions_per_reef, POPULATION_SIZE);
    }

    double *positions = opt->population[0].position;
#ifdef NO_OPENMP
    init_rng((unsigned int)time(NULL));
    double *rand_buffer = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
    if (!rand_buffer) {
        fprintf(stderr, "Failed to allocate random buffer\n");
        return;
    }

    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            int idx = i * solutions_per_reef + j;
            double *pos = positions + idx * opt->dim;
            for (int k = 0; k < opt->dim; k += 4) {
                if (k + 4 <= opt->dim) {
#ifdef __AVX2__
                    double lower[4], upper[4];
                    for (int m = 0; m < 4; m++) {
                        lower[m] = opt->bounds[2 * (k + m)];
                        upper[m] = opt->bounds[2 * (k + m) + 1];
                    }
                    __m256d lower_vec = _mm256_loadu_pd(lower);
                    __m256d upper_vec = _mm256_loadu_pd(upper);
                    generate_random_doubles(rand_buffer + k, 4, 0.0, 1.0);
                    __m256d rand_vec = _mm256_loadu_pd(rand_buffer + k);
                    __m256d result = _mm256_mul_pd(rand_vec, _mm256_sub_pd(upper_vec, lower_vec));
                    result = _mm256_add_pd(result, lower_vec);
                    _mm256_storeu_pd(pos + k, result);
#else
                    for (int m = 0; m < 4; m++) {
                        double lower = opt->bounds[2 * (k + m)];
                        double upper = opt->bounds[2 * (k + m) + 1];
                        pos[k + m] = RAND_DOUBLE(lower, upper);
                    }
#endif
                } else {
                    for (int m = k; m < opt->dim; m++) {
                        double lower = opt->bounds[2 * m];
                        double upper = opt->bounds[2 * m + 1];
                        pos[m] = RAND_DOUBLE(lower, upper);
                    }
                }
            }
            opt->population[idx].fitness = INFINITY;
        }
    }
    _mm_free(rand_buffer);
#else
    #pragma omp parallel
    {
        init_rng((unsigned int)time(NULL));
        double *rand_buffer = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
        if (!rand_buffer) {
            fprintf(stderr, "Failed to allocate random buffer\n");
            #pragma omp cancel parallel
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_REEFS; i++) {
            for (int j = 0; j < solutions_per_reef; j++) {
                int idx = i * solutions_per_reef + j;
                double *pos = positions + idx * opt->dim;
                for (int k = 0; k < opt->dim; k += 4) {
                    if (k + 4 <= opt->dim) {
#ifdef __AVX2__
                        double lower[4], upper[4];
                        for (int m = 0; m < 4; m++) {
                            lower[m] = opt->bounds[2 * (k + m)];
                            upper[m] = opt->bounds[2 * (k + m) + 1];
                        }
                        __m256d lower_vec = _mm256_loadu_pd(lower);
                        __m256d upper_vec = _mm256_loadu_pd(upper);
                        generate_random_doubles(rand_buffer + k, 4, 0.0, 1.0);
                        __m256d rand_vec = _mm256_loadu_pd(rand_buffer + k);
                        __m256d result = _mm256_mul_pd(rand_vec, _mm256_sub_pd(upper_vec, lower_vec));
                        result = _mm256_add_pd(result, lower_vec);
                        _mm256_storeu_pd(pos + k, result);
#else
                        for (int m = 0; m < 4; m++) {
                            double lower = opt->bounds[2 * (k + m)];
                            double upper = opt->bounds[2 * (k + m) + 1];
                            pos[k + m] = RAND_DOUBLE(lower, upper);
                        }
#endif
                    } else {
                        for (int m = k; m < opt->dim; m++) {
                            double lower = opt->bounds[2 * m];
                            double upper = opt->bounds[2 * m + 1];
                            pos[m] = RAND_DOUBLE(lower, upper);
                        }
                    }
                }
                opt->population[idx].fitness = INFINITY;
            }
        }
        _mm_free(rand_buffer);
    }
#endif
    enforce_bound_constraints(opt);
}

// Evaluate fitness for modified solutions only
void evaluate_reefs(Optimizer *opt, double (*objective_function)(double *), int *modified, int modified_count) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "Invalid optimizer, population, or objective function\n");
        return;
    }
#ifdef NO_OPENMP
    for (int m = 0; m < modified_count; m++) {
        int idx = modified[m];
        if (idx < 0 || idx >= opt->population_size || !opt->population[idx].position) {
            fprintf(stderr, "Invalid position for solution %d\n", idx);
            continue;
        }
        opt->population[idx].fitness = objective_function(opt->population[idx].position);
    }
#else
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < modified_count; m++) {
        int idx = modified[m];
        if (idx < 0 || idx >= opt->population_size || !opt->population[idx].position) {
            fprintf(stderr, "Invalid position for solution %d\n", idx);
            continue;
        }
        opt->population[idx].fitness = objective_function(opt->population[idx].position);
    }
#endif
}

// Migration phase: selective exchanges
void migration_phase_cfo(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef) {
    if (!opt || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }

    double *positions = opt->population[0].position;
    int total_solutions = opt->population_size;
    for (int k = 0; k < NUM_REEFS; k++) {
        if (*modified_count >= total_solutions) {
            fprintf(stderr, "Warning: modified array full in migration_phase_cfo\n");
            break;
        }
        int i = (int)(RAND_DOUBLE(0, NUM_REEFS));
        int j = (int)(RAND_DOUBLE(0, NUM_REEFS));
        if (i != j) {
            int idx = i * solutions_per_reef + ((int)RAND_DOUBLE(0, solutions_per_reef));
            int idx_replace = j * solutions_per_reef + ((int)RAND_DOUBLE(0, solutions_per_reef));
            if (idx >= total_solutions || idx_replace >= total_solutions) {
                fprintf(stderr, "Invalid index in migration_phase_cfo: idx=%d, idx_replace=%d\n", idx, idx_replace);
                continue;
            }
            double *src = positions + idx * opt->dim;
            double *dst = positions + idx_replace * opt->dim;
            memcpy(dst, src, opt->dim * sizeof(double));
            opt->population[idx_replace].fitness = INFINITY;
            modified[*modified_count] = idx_replace;
            (*modified_count)++;
        }
    }
}

// Local search phase: perturb solutions
void local_search_phase(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef) {
    if (!opt || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }

    double *positions = opt->population[0].position;
    int total_solutions = opt->population_size;
#ifdef NO_OPENMP
    double *rand_buffer = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
    if (!rand_buffer) {
        fprintf(stderr, "Failed to allocate random buffer\n");
        return;
    }

    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            if (*modified_count >= total_solutions) {
                fprintf(stderr, "Warning: modified array full in local_search_phase\n");
                break;
            }
            int idx = i * solutions_per_reef + j;
            if (idx >= total_solutions) {
                fprintf(stderr, "Invalid index in local_search_phase: idx=%d\n", idx);
                continue;
            }
            double *pos = positions + idx * opt->dim;
            generate_random_doubles(rand_buffer, opt->dim, -1.0, 1.0);
            for (int k = 0; k < opt->dim; k += 4) {
                if (k + 4 <= opt->dim) {
#ifdef __AVX2__
                    __m256d pos_vec = _mm256_loadu_pd(pos + k);
                    __m256d rand_vec = _mm256_loadu_pd(rand_buffer + k);
                    __m256d alpha_vec = _mm256_set1_pd(ALPHA * 2.0);
                    __m256d result = _mm256_fmadd_pd(rand_vec, alpha_vec, pos_vec);
                    _mm256_storeu_pd(pos + k, result);
#else
                    for (int m = 0; m < 4; m++) {
                        pos[k + m] += ALPHA * 2.0 * rand_buffer[k + m];
                    }
#endif
                } else {
                    for (int m = k; m < opt->dim; m++) {
                        pos[m] += ALPHA * 2.0 * rand_buffer[m];
                    }
                }
            }
            opt->population[idx].fitness = INFINITY;
            modified[*modified_count] = idx;
            (*modified_count)++;
        }
    }
    _mm_free(rand_buffer);
#else
    #pragma omp parallel
    {
        double *rand_buffer = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
        if (!rand_buffer) {
            fprintf(stderr, "Failed to allocate random buffer\n");
            #pragma omp cancel parallel
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_REEFS; i++) {
            for (int j = 0; j < solutions_per_reef; j++) {
                if (*modified_count >= total_solutions) {
                    fprintf(stderr, "Warning: modified array full in local_search_phase\n");
                    break;
                }
                int idx = i * solutions_per_reef + j;
                if (idx >= total_solutions) {
                    fprintf(stderr, "Invalid index in local_search_phase: idx=%d\n", idx);
                    continue;
                }
                double *pos = positions + idx * opt->dim;
                generate_random_doubles(rand_buffer, opt->dim, -1.0, 1.0);
                for (int k = 0; k < opt->dim; k += 4) {
                    if (k + 4 <= opt->dim) {
#ifdef __AVX2__
                        __m256d pos_vec = _mm256_loadu_pd(pos + k);
                        __m256d rand_vec = _mm256_loadu_pd(rand_buffer + k);
                        __m256d alpha_vec = _mm256_set1_pd(ALPHA * 2.0);
                        __m256d result = _mm256_fmadd_pd(rand_vec, alpha_vec, pos_vec);
                        _mm256_storeu_pd(pos + k, result);
#else
                        for (int m = 0; m < 4; m++) {
                            pos[k + m] += ALPHA * 2.0 * rand_buffer[k + m];
                        }
#endif
                    } else {
                        for (int m = k; m < opt->dim; m++) {
                            pos[m] += ALPHA * 2.0 * rand_buffer[m];
                        }
                    }
                }
                opt->population[idx].fitness = INFINITY;
                #pragma omp critical
                {
                    if (*modified_count < total_solutions) {
                        modified[*modified_count] = idx;
                        (*modified_count)++;
                    }
                }
            }
        }
        _mm_free(rand_buffer);
    }
#endif
}

// Main optimization function
void CRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Invalid optimizer or objective function\n");
        return;
    }

    // Initialize RNG
    init_rng((unsigned int)time(NULL));

    // Initialize reefs
    initialize_reefs(opt);
    if (!opt->population || !opt->population[0].position) {
        fprintf(stderr, "Initialization failed\n");
        return;
    }

    int solutions_per_reef = opt->population_size / NUM_REEFS;
    int total_solutions = opt->population_size;

    // Allocate modified solutions array (cache-aligned)
    int *modified = (int *)_mm_malloc(total_solutions * sizeof(int), 32);
    if (!modified) {
        fprintf(stderr, "Failed to allocate modified array\n");
        return;
    }

    // Evaluate initial fitness
    for (int i = 0; i < total_solutions; i++) {
        modified[i] = i;
    }
    evaluate_reefs(opt, objective_function, modified, total_solutions);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        int modified_count = 0;

        // Migration phase
        migration_phase_cfo(opt, modified, &modified_count, solutions_per_reef);

        // Local search phase
        local_search_phase(opt, modified, &modified_count, solutions_per_reef);

        // Evaluate modified solutions
        evaluate_reefs(opt, objective_function, modified, modified_count);

        // Find best solution (SIMD for large populations)
        double min_fitness = INFINITY;
        int best_idx = 0;
        double *positions = opt->population[0].position;
#ifdef __AVX2__
        __m256d min_vec = _mm256_set1_pd(INFINITY);
        int best_indices[4] = {0, 0, 0, 0};
        for (int i = 0; i < total_solutions - 3; i += 4) {
            __m256d fitness_vec = _mm256_set_pd(
                opt->population[i+3].fitness, opt->population[i+2].fitness,
                opt->population[i+1].fitness, opt->population[i].fitness
            );
            __m256d cmp = _mm256_cmp_pd(fitness_vec, min_vec, _CMP_LT_OQ);
            if (_mm256_movemask_pd(cmp)) {
                double fitness[4];
                _mm256_storeu_pd(fitness, fitness_vec);
                for (int j = 0; j < 4; j++) {
                    if (fitness[j] < min_fitness) {
                        min_fitness = fitness[j];
                        best_idx = i + j;
                    }
                }
                min_vec = _mm256_set1_pd(min_fitness);
            }
        }
        for (int i = (total_solutions / 4) * 4; i < total_solutions; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                best_idx = i;
            }
        }
#else
        for (int i = 0; i < total_solutions; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                best_idx = i;
            }
        }
#endif
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            if (!opt->best_solution.position) {
                fprintf(stderr, "Best solution position is null\n");
                _mm_free(modified);
                return;
            }
            memcpy(opt->best_solution.position, positions + best_idx * opt->dim, opt->dim * sizeof(double));
        }

        // Enforce bounds once per iteration
        enforce_bound_constraints(opt);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    _mm_free(modified);
}
