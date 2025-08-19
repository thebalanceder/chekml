#include "WOA.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <time.h> // For time()
#include <omp.h>

// Fast xorshift128+ random number generator
void woa_init_rng(woa_rng_state_t *state, uint64_t seed) {
    state->state[0] = seed ^ 0xdeadbeef;
    state->state[1] = seed ^ 0xcafebabe;
    // Warm up the RNG
    for (int i = 0; i < 10; i++) {
        woa_rand_double(state);
    }
}

double woa_rand_double(woa_rng_state_t *state) {
    uint64_t x = state->state[0];
    uint64_t y = state->state[1];
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    state->state[0] = y;
    state->state[1] = x;
    uint64_t r = (x + y) >> 12; // 52-bit mantissa for double
    return (double)r / (double)(1ULL << 52);
}

// Fast exp approximation (from Cephes library, simplified)
double woa_fast_exp(double x) {
    if (x < -20.0) return 0.0;
    if (x > 20.0) return 1e9;
    double z = x * 1.4426950408889634; // x / ln(2)
    int n = (int)(z + (z >= 0 ? 0.5 : -0.5));
    z = x - n * 0.6931471805599453; // x - n * ln(2)
    double t = z * z;
    double p = z * (1.0 + t * (0.3333333333333333 + t * 0.041666666666666664));
    return (1.0 + p) * (1ULL << n);
}

// Fast cos approximation (Taylor series, limited terms)
double woa_fast_cos(double x) {
    x = fmod(x, 2.0 * WOA_PI);
    if (x < 0) x += 2.0 * WOA_PI;
    double x2 = x * x;
    double result = 1.0 - x2 * 0.5 + x2 * x2 * 0.041666666666666664;
    return result;
}

// Initialize the population of search agents
void initialize_positions(Optimizer *opt, woa_rng_state_t *rng_states) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        woa_rng_state_t *rng = &rng_states[tid];
        #pragma omp for
        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                double lb = opt->bounds[2 * j];
                double ub = opt->bounds[2 * j + 1];
                pos[j] = lb + woa_rand_double(rng) * (ub - lb);
            }
            opt->population[i].fitness = INFINITY;
        }
    }
    enforce_bound_constraints(opt);
}

// Update the leader (best solution)
void update_leader(Optimizer *opt, double (*objective_function)(double *)) {
    double best_fitness = opt->best_solution.fitness;
    int best_idx = -1;

    #pragma omp parallel
    {
        double local_best_fitness = INFINITY;
        int local_best_idx = -1;
        #pragma omp for
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < local_best_fitness) {
                local_best_fitness = fitness;
                local_best_idx = i;
            }
        }
        #pragma omp critical
        {
            if (local_best_fitness < best_fitness) {
                best_fitness = local_best_fitness;
                best_idx = local_best_idx;
            }
        }
    }

    if (best_idx >= 0) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
    }
}

// Update positions of search agents (AVX2 vectorized for dim >= 4)
void update_positions(Optimizer *opt, int t, woa_rng_state_t *rng_states) {
    double max_iter = (double)opt->max_iter;
    double a = WOA_A_INITIAL - t * (WOA_A_INITIAL / max_iter);
    double a2 = WOA_A2_INITIAL + t * ((WOA_A2_FINAL - WOA_A2_INITIAL) / max_iter);
    double b = WOA_B;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        woa_rng_state_t *rng = &rng_states[tid];
        #pragma omp for
        for (int i = 0; i < opt->population_size; i++) {
            double r1 = woa_rand_double(rng);
            double r2 = woa_rand_double(rng);
            double A = 2.0 * a * r1 - a;  // Eq. (2.3)
            double C = 2.0 * r2;          // Eq. (2.4)
            double l = (a2 - 1.0) * woa_rand_double(rng) + 1.0;  // Eq. (2.5)
            double p = woa_rand_double(rng);  // Strategy selection
            double *pos = opt->population[i].position;
            double *best_pos = opt->best_solution.position;

            if (opt->dim >= 4 && p < 0.5 && fabs(A) < 1.0) {  // Vectorized encircling prey
                __m256d A_vec = _mm256_set1_pd(A);
                __m256d C_vec = _mm256_set1_pd(C);
                for (int j = 0; j <= opt->dim - 4; j += 4) {
                    __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                    __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
                    __m256d diff = _mm256_sub_pd(best_vec, pos_vec);
                    __m256d D_Leader = _mm256_mul_pd(C_vec, diff);
                    __m256d new_pos = _mm256_sub_pd(best_vec, _mm256_mul_pd(A_vec, D_Leader));
                    _mm256_storeu_pd(&pos[j], new_pos);
                }
                // Handle remaining dimensions
                for (int j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                    double D_Leader = fabs(C * best_pos[j] - pos[j]);
                    pos[j] = best_pos[j] - A * D_Leader;
                }
            } else if (opt->dim >= 4 && p < 0.5 && fabs(A) >= 1.0) {  // Vectorized search for prey
                __m256d A_vec = _mm256_set1_pd(A);
                __m256d C_vec = _mm256_set1_pd(C);
                int rand_idx = (int)(woa_rand_double(rng) * opt->population_size);
                double *rand_pos = opt->population[rand_idx].position;
                for (int j = 0; j <= opt->dim - 4; j += 4) {
                    __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                    __m256d rand_vec = _mm256_loadu_pd(&rand_pos[j]);
                    __m256d diff = _mm256_sub_pd(_mm256_mul_pd(C_vec, rand_vec), pos_vec);
                    __m256d D_X_rand = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // fabs
                    __m256d new_pos = _mm256_sub_pd(rand_vec, _mm256_mul_pd(A_vec, D_X_rand));
                    _mm256_storeu_pd(&pos[j], new_pos);
                }
                for (int j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                    double D_X_rand = fabs(C * rand_pos[j] - pos[j]);
                    pos[j] = rand_pos[j] - A * D_X_rand;
                }
            } else if (opt->dim >= 4 && p >= 0.5) {  // Vectorized spiral update
                double spiral_factor = woa_fast_exp(b * l) * woa_fast_cos(l * 2.0 * WOA_PI);
                __m256d factor_vec = _mm256_set1_pd(spiral_factor);
                for (int j = 0; j <= opt->dim - 4; j += 4) {
                    __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                    __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
                    __m256d diff = _mm256_sub_pd(best_vec, pos_vec);
                    __m256d distance = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // fabs
                    __m256d new_pos = _mm256_add_pd(_mm256_mul_pd(distance, factor_vec), best_vec);
                    _mm256_storeu_pd(&pos[j], new_pos);
                }
                for (int j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                    double distance2Leader = fabs(best_pos[j] - pos[j]);
                    pos[j] = distance2Leader * spiral_factor + best_pos[j];
                }
            } else {  // Scalar fallback for small dim or edge cases
                for (int j = 0; j < opt->dim; j++) {
                    if (p < 0.5) {
                        if (fabs(A) >= 1.0) {
                            int rand_idx = (int)(woa_rand_double(rng) * opt->population_size);
                            double X_rand_j = opt->population[rand_idx].position[j];
                            double D_X_rand = fabs(C * X_rand_j - pos[j]);
                            pos[j] = X_rand_j - A * D_X_rand;
                        } else {
                            double D_Leader = fabs(C * best_pos[j] - pos[j]);
                            pos[j] = best_pos[j] - A * D_Leader;
                        }
                    } else {
                        double distance2Leader = fabs(best_pos[j] - pos[j]);
                        pos[j] = distance2Leader * woa_fast_exp(b * l) * woa_fast_cos(l * 2.0 * WOA_PI) + best_pos[j];
                    }
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize per-thread RNG states
    int nthreads = omp_get_max_threads();
    woa_rng_state_t *rng_states = (woa_rng_state_t *)malloc(nthreads * sizeof(woa_rng_state_t));
    uint64_t seed = (uint64_t)time(NULL);
    for (int i = 0; i < nthreads; i++) {
        woa_init_rng(&rng_states[i], seed + i);
    }

    // Initialize positions
    initialize_positions(opt, rng_states);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        update_leader(opt, objective_function);
        update_positions(opt, t, rng_states);
    }

    free(rng_states);
}
