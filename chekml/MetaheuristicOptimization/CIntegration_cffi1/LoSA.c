#include "LoSA.h"
#include "generaloptimizer.h"
#include <immintrin.h>  // For AVX2 intrinsics
#include <string.h>     // For memcpy
#include <time.h>       // For time()
#ifdef _OPENMP
#include <omp.h>        // For parallelization
#endif

// Initialize Population
void LoSA_initialize_population(Optimizer *opt, LoSA_XorshiftState *rng) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *bounds = opt->bounds;

    // Allocate aligned temporary buffer for positions
    double *positions = (double *)aligned_malloc(pop_size * dim * sizeof(double), 32);
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            positions[i * dim + j] = lower + (upper - lower) * LoSA_xorshift_double(rng);
        }
    }

    // Copy to population and set fitness
    #pragma omp parallel for
    for (int i = 0; i < pop_size; i++) {
        memcpy(opt->population[i].position, &positions[i * dim], dim * sizeof(double));
        opt->population[i].fitness = INFINITY;
    }

    aligned_free(positions);
}

// Update Positions (AVX2 vectorized)
void LoSA_update_positions(Optimizer *opt, LoSA_XorshiftState *rng, double *rand_buffer) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_pos = opt->best_solution.position;
    const __m256d step_size_vec = _mm256_set1_pd(LOSA_STEP_SIZE);

    // Generate random numbers for all positions
    for (int i = 0; i < pop_size * dim; i++) {
        rand_buffer[i] = LoSA_xorshift_double(rng);
    }

    // Vectorized position updates
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int j = 0;
        for (; j <= dim - LOSA_SIMD_WIDTH; j += LOSA_SIMD_WIDTH) {
            __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
            __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
            __m256d rand_vec = _mm256_loadu_pd(&rand_buffer[i * dim + j]);
            __m256d direction = _mm256_sub_pd(best_vec, pos_vec);
            __m256d update = _mm256_mul_pd(_mm256_mul_pd(step_size_vec, direction), rand_vec);
            pos_vec = _mm256_add_pd(pos_vec, update);
            _mm256_storeu_pd(&pos[j], pos_vec);
        }
        // Handle remaining dimensions
        for (; j < dim; j++) {
            double direction = best_pos[j] - pos[j];
            pos[j] += LOSA_STEP_SIZE * direction * rand_buffer[i * dim + j];
        }

        // Inline bound clamping
        for (j = 0; j < dim; j++) {
            double lower = opt->bounds[2 * j];
            double upper = opt->bounds[2 * j + 1];
            pos[j] = fmax(lower, fmin(upper, pos[j]));
        }
    }
}

// Main Optimization Function
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int max_iter = opt->max_iter;
    double *best_pos = opt->best_solution.position;
    double *best_fitness = &opt->best_solution.fitness;

    // Initialize random number generator
    LoSA_XorshiftState rng;
    LoSA_xorshift_init(&rng, (uint64_t)time(NULL));

    // Preallocate random number buffer
    double *rand_buffer = (double *)aligned_malloc(pop_size * dim * sizeof(double), 32);

    // Initialize population
    LoSA_initialize_population(opt, &rng);
    *best_fitness = INFINITY;

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness and update best solution (parallelized)
        #pragma omp parallel
        {
            double local_best_fitness = INFINITY;
            double *local_best_pos = (double *)aligned_malloc(dim * sizeof(double), 32);
            int found_better = 0;

            #pragma omp for schedule(guided)
            for (int i = 0; i < pop_size; i++) {
                double new_fitness = objective_function(opt->population[i].position);
                opt->population[i].fitness = new_fitness;
                if (new_fitness < local_best_fitness) {
                    local_best_fitness = new_fitness;
                    memcpy(local_best_pos, opt->population[i].position, dim * sizeof(double));
                    found_better = 1;
                }
            }

            #pragma omp critical
            {
                if (found_better && local_best_fitness < *best_fitness) {
                    *best_fitness = local_best_fitness;
                    memcpy(best_pos, local_best_pos, dim * sizeof(double));
                }
            }

            aligned_free(local_best_pos);
        }

        // Update positions
        LoSA_update_positions(opt, &rng, rand_buffer);

        // Log progress (optional, can be disabled for max speed)
        printf("Iteration %d: Best Value = %f\n", iter + 1, *best_fitness);
    }

    aligned_free(rand_buffer);
}
