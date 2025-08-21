#include "MSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <emmintrin.h> // SSE2
#include <immintrin.h> // AVX

// Initialize Xorshift128+ RNG
void mso_xorshift_init(MSO_XorshiftState *state, unsigned long long seed) {
    state->s[0] = seed | 1; // Ensure non-zero
    state->s[1] = seed ^ 0xDEADBEEF;
    for (int i = 0; i < 8; i++) { // Warm up RNG
        mso_xorshift_next(state);
    }
}

// Generate next Xorshift128+ random number
static inline unsigned long long mso_xorshift_next(MSO_XorshiftState *state) {
    unsigned long long x = state->s[0];
    unsigned long long y = state->s[1];
    state->s[0] = y;
    x ^= x << 23;
    state->s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return state->s[1] + y;
}

// Get uniform random double from buffer (refill if empty)
static inline double mso_xorshift_double(MSO_RNGBuffer *buffer, MSO_XorshiftState *state) {
    if (buffer->uniform_idx >= buffer->size) {
        for (size_t i = 0; i < buffer->size; i++) {
            buffer->uniform[i] = (double)mso_xorshift_next(state) / (1ULL << 32) / (1ULL << 32);
        }
        buffer->uniform_idx = 0;
    }
    return buffer->uniform[buffer->uniform_idx++];
}

// Ziggurat algorithm for normal distribution (simplified)
static inline double mso_xorshift_normal(MSO_RNGBuffer *buffer, MSO_XorshiftState *state) {
    if (buffer->normal_idx >= buffer->size) {
        // Precompute normal random numbers using Box-Muller (Ziggurat tables omitted for simplicity)
        for (size_t i = 0; i < buffer->size; i += 2) {
            double u1 = mso_xorshift_double(buffer, state);
            double u2 = mso_xorshift_double(buffer, state);
            double r = sqrt(-2.0 * log(u1));
            double theta = 2.0 * M_PI * u2;
            buffer->normal[i] = r * cos(theta);
            if (i + 1 < buffer->size) {
                buffer->normal[i + 1] = r * sin(theta);
            }
        }
        buffer->normal_idx = 0;
    }
    return buffer->normal[buffer->normal_idx++];
}

// Initialize RNG buffer
void mso_rng_buffer_init(MSO_RNGBuffer *buffer, MSO_XorshiftState *state, size_t size) {
    buffer->uniform = (double *)_mm_malloc(size * sizeof(double), 32); // AVX alignment
    buffer->normal = (double *)_mm_malloc(size * sizeof(double), 32);
    buffer->size = size;
    buffer->uniform_idx = size; // Force initial fill
    buffer->normal_idx = size;
    for (size_t i = 0; i < size; i++) {
        buffer->uniform[i] = mso_xorshift_double(buffer, state);
        buffer->normal[i] = mso_xorshift_normal(buffer, state);
    }
    buffer->uniform_idx = 0;
    buffer->normal_idx = 0;
}

// Free RNG buffer
void mso_rng_buffer_free(MSO_RNGBuffer *buffer) {
    _mm_free(buffer->uniform);
    _mm_free(buffer->normal);
    buffer->uniform = NULL;
    buffer->normal = NULL;
    buffer->size = 0;
}

// Update positions (parallelized, vectorized, unrolled)
void mso_update_positions(Optimizer *opt, int iter, MSO_XorshiftState *rng_state, MSO_RNGBuffer *rng_buffer) {
    double p_explore = fmax(MSO_MAX_P_EXPLORE * exp(-0.1 * iter), MSO_MIN_P_EXPLORE);
    double *best_pos = opt->best_solution.position;
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    #pragma omp parallel
    {
        MSO_XorshiftState local_rng = *rng_state; // Thread-local RNG
        MSO_RNGBuffer local_buffer;
        mso_rng_buffer_init(&local_buffer, &local_rng, MSO_RNG_BUFFER_SIZE);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < pop_size; i++) {
            double *pos = opt->population[i].position;

            // Unroll for dim=2 (common case)
            if (dim == 2) {
                double rand_val = mso_xorshift_double(&local_buffer, &local_rng);
                if (rand_val < p_explore) { // Exploration
                    double r = mso_xorshift_double(&local_buffer, &local_rng);
                    if (r < 0.5) {
                        pos[0] = best_pos[0] + mso_xorshift_double(&local_buffer, &local_rng) * (bounds[1] - best_pos[0]);
                        pos[1] = best_pos[1] + mso_xorshift_double(&local_buffer, &local_rng) * (bounds[3] - best_pos[1]);
                    } else {
                        pos[0] = best_pos[0] - mso_xorshift_double(&local_buffer, &local_rng) * (best_pos[0] - bounds[0]);
                        pos[1] = best_pos[1] - mso_xorshift_double(&local_buffer, &local_rng) * (best_pos[1] - bounds[2]);
                    }
                } else { // Exploitation
                    pos[0] = best_pos[0] + mso_xorshift_normal(&local_buffer, &local_rng) * (bounds[1] - bounds[0]) / MSO_PERTURBATION_SCALE;
                    pos[1] = best_pos[1] + mso_xorshift_normal(&local_buffer, &local_rng) * (bounds[3] - bounds[2]) / MSO_PERTURBATION_SCALE;
                }
            } else {
                // General case with vectorization for large dim
                for (int j = 0; j < dim; j += 4) { // Process 4 elements at a time (AVX)
                    if (j + 4 <= dim) {
                        __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                        __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
                        __m256d bound_lo = _mm256_loadu_pd(&bounds[2 * j]);
                        __m256d bound_hi = _mm256_loadu_pd(&bounds[2 * j + 1]);
                        __m256d range = _mm256_sub_pd(bound_hi, bound_lo);

                        double rand_val = mso_xorshift_double(&local_buffer, &local_rng);
                        if (rand_val < p_explore) { // Exploration
                            double r = mso_xorshift_double(&local_buffer, &local_rng);
                            __m256d rand_vec = _mm256_set1_pd(mso_xorshift_double(&local_buffer, &local_rng));
                            if (r < 0.5) {
                                __m256d delta = _mm256_sub_pd(bound_hi, best_vec);
                                pos_vec = _mm256_add_pd(best_vec, _mm256_mul_pd(rand_vec, delta));
                            } else {
                                __m256d delta = _mm256_sub_pd(best_vec, bound_lo);
                                pos_vec = _mm256_sub_pd(best_vec, _mm256_mul_pd(rand_vec, delta));
                            }
                        } else { // Exploitation
                            __m256d normal_vec = _mm256_set_pd(
                                mso_xorshift_normal(&local_buffer, &local_rng),
                                mso_xorshift_normal(&local_buffer, &local_rng),
                                mso_xorshift_normal(&local_buffer, &local_rng),
                                mso_xorshift_normal(&local_buffer, &local_rng)
                            );
                            __m256d scale = _mm256_div_pd(range, _mm256_set1_pd(MSO_PERTURBATION_SCALE));
                            pos_vec = _mm256_add_pd(best_vec, _mm256_mul_pd(normal_vec, scale));
                        }
                        _mm256_storeu_pd(&pos[j], pos_vec);
                    } else {
                        // Handle remaining elements
                        for (; j < dim; j++) {
                            double rand_val = mso_xorshift_double(&local_buffer, &local_rng);
                            if (rand_val < p_explore) {
                                double r = mso_xorshift_double(&local_buffer, &local_rng);
                                if (r < 0.5) {
                                    pos[j] = best_pos[j] + mso_xorshift_double(&local_buffer, &local_rng) * (bounds[2 * j + 1] - best_pos[j]);
                                } else {
                                    pos[j] = best_pos[j] - mso_xorshift_double(&local_buffer, &local_rng) * (best_pos[j] - bounds[2 * j]);
                                }
                            } else {
                                pos[j] = best_pos[j] + mso_xorshift_normal(&local_buffer, &local_rng) * (bounds[2 * j + 1] - bounds[2 * j]) / MSO_PERTURBATION_SCALE;
                            }
                        }
                    }
                }
            }
        }
        mso_rng_buffer_free(&local_buffer);
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    MSO_XorshiftState rng_state;
    mso_xorshift_init(&rng_state, (unsigned long long)time(NULL));
    MSO_RNGBuffer rng_buffer;
    mso_rng_buffer_init(&rng_buffer, &rng_state, MSO_RNG_BUFFER_SIZE);

    // Initialize population fitness (parallelized)
    double *best_pos = opt->best_solution.position;
    double best_fitness = INFINITY;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    #pragma omp parallel
    {
        double local_best_fitness = INFINITY;
        double *local_best_pos = (double *)_mm_malloc(dim * sizeof(double), 32);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < pop_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < local_best_fitness) {
                local_best_fitness = fitness;
                memcpy(local_best_pos, opt->population[i].position, dim * sizeof(double));
            }
        }

        #pragma omp critical
        {
            if (local_best_fitness < best_fitness) {
                best_fitness = local_best_fitness;
                memcpy(best_pos, local_best_pos, dim * sizeof(double));
            }
        }
        _mm_free(local_best_pos);
    }
    opt->best_solution.fitness = best_fitness;

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        mso_update_positions(opt, iter, &rng_state, &rng_buffer);

        // Evaluate fitness and update best solution (parallelized)
        best_fitness = opt->best_solution.fitness;
        #pragma omp parallel
        {
            double local_best_fitness = INFINITY;
            double *local_best_pos = (double *)_mm_malloc(dim * sizeof(double), 32);

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < pop_size; i++) {
                double *pos = opt->population[i].position;
                double fitness = objective_function(pos);
                opt->population[i].fitness = fitness;
                if (fitness < local_best_fitness) {
                    local_best_fitness = fitness;
                    memcpy(local_best_pos, pos, dim * sizeof(double));
                }
            }

            #pragma omp critical
            {
                if (local_best_fitness < best_fitness) {
                    best_fitness = local_best_fitness;
                    memcpy(best_pos, local_best_pos, dim * sizeof(double));
                }
            }
            _mm_free(local_best_pos);
        }
        opt->best_solution.fitness = best_fitness;

        // Optional: Print iteration progress
        #pragma omp single
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, best_fitness);
    }

    mso_rng_buffer_free(&rng_buffer);
}
