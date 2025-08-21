#include "HHO.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Fast PCG32 RNG (period 2^32)
static inline uint32_t pcg32(uint32_t *state) {
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Optimized random double in [min, max)
static inline double hho_rand_double(double min, double max, uint32_t *rng_state) {
    return min + (max - min) * (pcg32(rng_state) / 4294967296.0);
}

// Fast log approximation (for x > 0, near 1)
static inline double fast_log(double x) {
    // Pade approximation: log(x) ≈ (x - 1) * (x + 1) / (x + 0.5)
    double y = x - 1.0;
    return y * (x + 1.0) / (x + 0.5);
}

// Fast cos approximation (for x in [0, 2pi])
static inline double fast_cos(double x) {
    // Taylor series approximation: cos(x) ≈ 1 - x^2/2 + x^4/24
    x = fmod(x, 2.0 * M_PI); // Reduce to [0, 2pi]
    double x2 = x * x;
    return 1.0 - x2 / 2.0 + x2 * x2 / 24.0;
}

// Clamp position to bounds
static inline void clamp_position(double *pos, int dim, const double *bounds) {
    for (int j = 0; j < dim; j++) {
        if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
        else if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
    }
}

// Optimized Levy flight with fast approximations and stability
void levy_flight(double *step, int dim, uint32_t *rng_state) {
    __m256d sigma = _mm256_set1_pd(LEVY_SIGMA);
    __m256d one_over_beta = _mm256_set1_pd(1.0 / HHO_BETA);
    __m256d two = _mm256_set1_pd(2.0);
    __m256d two_pi = _mm256_set1_pd(2.0 * M_PI);

    for (int i = 0; i < dim; i += 4) {
        // Generate random numbers with bounds checking
        double r1[4], r2[4], v[4];
        for (int j = 0; j < 4 && i + j < dim; j++) {
            r1[j] = hho_rand_double(0.0, 1.0, rng_state);
            r2[j] = hho_rand_double(0.0, 1.0, rng_state);
            v[j] = hho_rand_double(0.0, 1.0, rng_state);
            // Avoid log(0) and negative inputs
            r1[j] = fmax(r1[j], 1e-10);
            v[j] = fmax(v[j], 1e-10);
        }

        // Box-Muller transform with fast approximations
        double z0[4];
        for (int j = 0; j < 4 && i + j < dim; j++) {
            double sqrt_term = sqrt(-2.0 * fast_log(r1[j]));
            z0[j] = sqrt_term * fast_cos(two_pi[0] * r2[j]);
        }

        // Levy step
        __m256d z0_vec = _mm256_loadu_pd(z0);
        __m256d v_vec = _mm256_loadu_pd(v);
        __m256d abs_v = _mm256_max_pd(v_vec, _mm256_sub_pd(_mm256_setzero_pd(), v_vec));
        
        // Compute pow(abs_v, one_over_beta) with fast approximation
        double abs_v_array[4], pow_result[4];
        _mm256_storeu_pd(abs_v_array, abs_v);
        for (int j = 0; j < 4 && i + j < dim; j++) {
            // Approximate pow(x, 1/1.5) = x^(2/3)
            pow_result[j] = cbrt(abs_v_array[j] * abs_v_array[j]);
        }
        __m256d pow_vec = _mm256_loadu_pd(pow_result);

        __m256d levy = _mm256_div_pd(_mm256_mul_pd(sigma, z0_vec), pow_vec);

        // Store results
        double temp[4];
        _mm256_storeu_pd(temp, levy);
        for (int j = 0; j < 4 && i + j < dim; j++) {
            step[i + j] = temp[j];
        }
    }
}

// Exploration Phase
void exploration_phase(Optimizer *opt, double (*objective_function)(double *)) {
    uint32_t rng_state = (uint32_t)time(NULL);
    double mean_pos[opt->dim];
    __m256d zero = _mm256_setzero_pd();

    // Cache population mean
    memset(mean_pos, 0, opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        __m256d sum = zero;
        for (int k = 0; k < opt->population_size; k += 4) {
            _mm_prefetch((const char *)&opt->population[k + 4].position[j], _MM_HINT_T0);
            if (k + 4 <= opt->population_size) {
                sum = _mm256_add_pd(sum, _mm256_loadu_pd(&opt->population[k].position[j]));
            } else {
                for (int m = k; m < opt->population_size; m++) {
                    mean_pos[j] += opt->population[m].position[j];
                }
            }
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        for (int m = 0; m < 4; m++) mean_pos[j] += temp[m];
        mean_pos[j] /= opt->population_size;
    }

    #pragma omp parallel
    {
        uint32_t local_rng = rng_state + omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < opt->population_size; i++) {
            double q = hho_rand_double(0.0, 1.0, &local_rng);
            int rand_hawk_index = (int)(hho_rand_double(0.0, opt->population_size, &local_rng));

            if (q < 0.5) {
                // Perch based on other family members (vectorized)
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d r1 = _mm256_set1_pd(hho_rand_double(0.0, 1.0, &local_rng));
                        __m256d r2 = _mm256_set1_pd(hho_rand_double(0.0, 1.0, &local_rng));
                        __m256d hawk_pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                        __m256d rand_hawk = _mm256_loadu_pd(&opt->population[rand_hawk_index].position[j]);
                        __m256d diff = _mm256_sub_pd(rand_hawk, _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(r2, hawk_pos)));
                        __m256d abs_diff = _mm256_max_pd(diff, _mm256_sub_pd(zero, diff));
                        __m256d result = _mm256_sub_pd(rand_hawk, _mm256_mul_pd(r1, abs_diff));
                        _mm256_storeu_pd(&opt->population[i].position[j], result);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            double r1 = hho_rand_double(0.0, 1.0, &local_rng);
                            double r2 = hho_rand_double(0.0, 1.0, &local_rng);
                            opt->population[i].position[jj] = 
                                opt->population[rand_hawk_index].position[jj] - 
                                r1 * fabs(opt->population[rand_hawk_index].position[jj] - 2 * r2 * opt->population[i].position[jj]);
                        }
                    }
                }
            } else {
                // Perch on a random tall tree
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d r = _mm256_set1_pd(hho_rand_double(0.0, 1.0, &local_rng));
                        __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                        __m256d mean = _mm256_loadu_pd(&mean_pos[j]);
                        __m256d bound_diff = _mm256_sub_pd(
                            _mm256_loadu_pd(&opt->bounds[2 * j + 1]),
                            _mm256_loadu_pd(&opt->bounds[2 * j])
                        );
                        __m256d rand_term = _mm256_mul_pd(bound_diff, r);
                        __m256d result = _mm256_sub_pd(
                            _mm256_sub_pd(best_pos, mean),
                            _mm256_add_pd(rand_term, _mm256_loadu_pd(&opt->bounds[2 * j]))
                        );
                        _mm256_storeu_pd(&opt->population[i].position[j], result);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            double r = hho_rand_double(0.0, 1.0, &local_rng);
                            opt->population[i].position[jj] = 
                                (opt->best_solution.position[jj] - mean_pos[jj]) - 
                                r * ((opt->bounds[2 * jj + 1] - opt->bounds[2 * jj]) * hho_rand_double(0.0, 1.0, &local_rng) + opt->bounds[2 * jj]);
                        }
                    }
                }
            }
            clamp_position(opt->population[i].position, opt->dim, opt->bounds);
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *)) {
    uint32_t rng_state = (uint32_t)time(NULL);
    double levy_step[opt->dim], X1[opt->dim], X2[opt->dim], mean_pos[opt->dim];
    __m256d zero = _mm256_setzero_pd();

    // Cache population mean
    memset(mean_pos, 0, opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        __m256d sum = zero;
        for (int k = 0; k < opt->population_size; k += 4) {
            _mm_prefetch((const char *)&opt->population[k + 4].position[j], _MM_HINT_T0);
            if (k + 4 <= opt->population_size) {
                sum = _mm256_add_pd(sum, _mm256_loadu_pd(&opt->population[k].position[j]));
            } else {
                for (int m = k; m < opt->population_size; m++) {
                    mean_pos[j] += opt->population[m].position[j];
                }
            }
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        for (int m = 0; m < 4; m++) mean_pos[j] += temp[m];
        mean_pos[j] /= opt->population_size;
    }

    #pragma omp parallel
    {
        uint32_t local_rng = rng_state + omp_get_thread_num();
        double local_levy[opt->dim], local_X1[opt->dim], local_X2[opt->dim];
        #pragma omp for
        for (int i = 0; i < opt->population_size; i++) {
            double r = hho_rand_double(0.0, 1.0, &local_rng);
            double jump_strength = 2.0 * (1.0 - hho_rand_double(0.0, 1.0, &local_rng));
            double escaping_energy = opt->population[i].fitness;

            if (r >= 0.5 && fabs(escaping_energy) < 0.5) {
                // Hard besiege
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                        __m256d curr_pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                        __m256d diff = _mm256_sub_pd(best_pos, curr_pos);
                        __m256d abs_diff = _mm256_max_pd(diff, _mm256_sub_pd(zero, diff));
                        __m256d result = _mm256_sub_pd(best_pos, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_diff));
                        _mm256_storeu_pd(&opt->population[i].position[j], result);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            opt->population[i].position[jj] = 
                                opt->best_solution.position[jj] - 
                                escaping_energy * fabs(opt->best_solution.position[jj] - opt->population[i].position[jj]);
                        }
                    }
                }
                clamp_position(opt->population[i].position, opt->dim, opt->bounds);
            } else if (r >= 0.5 && fabs(escaping_energy) >= 0.5) {
                // Soft besiege
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                        __m256d curr_pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                        __m256d diff = _mm256_sub_pd(best_pos, curr_pos);
                        __m256d jump_diff = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(jump_strength), best_pos), curr_pos);
                        __m256d abs_jump = _mm256_max_pd(jump_diff, _mm256_sub_pd(zero, jump_diff));
                        __m256d result = _mm256_sub_pd(diff, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_jump));
                        _mm256_storeu_pd(&opt->population[i].position[j], result);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            opt->population[i].position[jj] = 
                                (opt->best_solution.position[jj] - opt->population[i].position[jj]) - 
                                escaping_energy * fabs(jump_strength * opt->best_solution.position[jj] - opt->population[i].position[jj]);
                        }
                    }
                }
                clamp_position(opt->population[i].position, opt->dim, opt->bounds);
            } else if (r < 0.5 && fabs(escaping_energy) >= 0.5) {
                // Soft besiege with rapid dives
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                        __m256d curr_pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                        __m256d jump_diff = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(jump_strength), best_pos), curr_pos);
                        __m256d abs_jump = _mm256_max_pd(jump_diff, _mm256_sub_pd(zero, jump_diff));
                        __m256d X1_vec = _mm256_sub_pd(best_pos, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_jump));
                        _mm256_storeu_pd(&local_X1[j], X1_vec);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            local_X1[jj] = opt->best_solution.position[jj] - 
                                           escaping_energy * fabs(jump_strength * opt->best_solution.position[jj] - opt->population[i].position[jj]);
                        }
                    }
                }
                clamp_position(local_X1, opt->dim, opt->bounds);
                double X1_fitness = objective_function(local_X1);
                
                if (X1_fitness < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, local_X1, opt->dim * sizeof(double));
                    opt->population[i].fitness = X1_fitness;
                } else {
                    levy_flight(local_levy, opt->dim, &local_rng);
                    for (int j = 0; j < opt->dim; j += 4) {
                        if (j + 4 <= opt->dim) {
                            __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                            __m256d curr_pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                            __m256d jump_diff = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(jump_strength), best_pos), curr_pos);
                            __m256d abs_jump = _mm256_max_pd(jump_diff, _mm256_sub_pd(zero, jump_diff));
                            __m256d levy_term = _mm256_mul_pd(_mm256_loadu_pd(&local_levy[j]), _mm256_set1_pd(hho_rand_double(0.0, 1.0, &local_rng)));
                            __m256d X2_vec = _mm256_add_pd(
                                _mm256_sub_pd(best_pos, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_jump)),
                                levy_term
                            );
                            _mm256_storeu_pd(&local_X2[j], X2_vec);
                        } else {
                            for (int jj = j; jj < opt->dim; jj++) {
                                local_X2[jj] = opt->best_solution.position[jj] - 
                                               escaping_energy * fabs(jump_strength * opt->best_solution.position[jj] - opt->population[i].position[jj]) + 
                                               hho_rand_double(0.0, 1.0, &local_rng) * local_levy[jj];
                            }
                        }
                    }
                    clamp_position(local_X2, opt->dim, opt->bounds);
                    double X2_fitness = objective_function(local_X2);
                    if (X2_fitness < opt->population[i].fitness) {
                        memcpy(opt->population[i].position, local_X2, opt->dim * sizeof(double));
                        opt->population[i].fitness = X2_fitness;
                    }
                }
            } else {
                // Hard besiege with rapid dives
                for (int j = 0; j < opt->dim; j += 4) {
                    if (j + 4 <= opt->dim) {
                        __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                        __m256d mean = _mm256_loadu_pd(&mean_pos[j]);
                        __m256d jump_diff = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(jump_strength), best_pos), mean);
                        __m256d abs_jump = _mm256_max_pd(jump_diff, _mm256_sub_pd(zero, jump_diff));
                        __m256d X1_vec = _mm256_sub_pd(best_pos, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_jump));
                        _mm256_storeu_pd(&local_X1[j], X1_vec);
                    } else {
                        for (int jj = j; jj < opt->dim; jj++) {
                            local_X1[jj] = opt->best_solution.position[jj] - 
                                           escaping_energy * fabs(jump_strength * opt->best_solution.position[jj] - mean_pos[jj]);
                        }
                    }
                }
                clamp_position(local_X1, opt->dim, opt->bounds);
                double X1_fitness = objective_function(local_X1);
                
                if (X1_fitness < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, local_X1, opt->dim * sizeof(double));
                    opt->population[i].fitness = X1_fitness;
                } else {
                    levy_flight(local_levy, opt->dim, &local_rng);
                    for (int j = 0; j < opt->dim; j += 4) {
                        if (j + 4 <= opt->dim) {
                            __m256d best_pos = _mm256_loadu_pd(&opt->best_solution.position[j]);
                            __m256d mean = _mm256_loadu_pd(&mean_pos[j]);
                            __m256d jump_diff = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(jump_strength), best_pos), mean);
                            __m256d abs_jump = _mm256_max_pd(jump_diff, _mm256_sub_pd(zero, jump_diff));
                            __m256d levy_term = _mm256_mul_pd(_mm256_loadu_pd(&local_levy[j]), _mm256_set1_pd(hho_rand_double(0.0, 1.0, &local_rng)));
                            __m256d X2_vec = _mm256_add_pd(
                                _mm256_sub_pd(best_pos, _mm256_mul_pd(_mm256_set1_pd(escaping_energy), abs_jump)),
                                levy_term
                            );
                            _mm256_storeu_pd(&local_X2[j], X2_vec);
                        } else {
                            for (int jj = j; jj < opt->dim; jj++) {
                                local_X2[jj] = opt->best_solution.position[jj] - 
                                               escaping_energy * fabs(jump_strength * opt->best_solution.position[jj] - mean_pos[jj]) + 
                                               hho_rand_double(0.0, 1.0, &local_rng) * local_levy[jj];
                            }
                        }
                    }
                    clamp_position(local_X2, opt->dim, opt->bounds);
                    double X2_fitness = objective_function(local_X2);
                    if (X2_fitness < opt->population[i].fitness) {
                        memcpy(opt->population[i].position, local_X2, opt->dim * sizeof(double));
                        opt->population[i].fitness = X2_fitness;
                    }
                }
            }
            clamp_position(opt->population[i].position, opt->dim, opt->bounds);
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    uint32_t rng_state = (uint32_t)time(NULL);

    for (int t = 0; t < opt->max_iter; t++) {
        // Update fitness and find best solution
        #pragma omp parallel
        {
            uint32_t local_rng = rng_state + omp_get_thread_num();
            double local_best_fitness = opt->best_solution.fitness;
            double local_best_pos[opt->dim];
            #pragma omp for
            for (int i = 0; i < opt->population_size; i++) {
                _mm_prefetch((const char *)&opt->population[i + 1].position[0], _MM_HINT_T0);
                double fitness = objective_function(opt->population[i].position);
                opt->population[i].fitness = fitness;
                if (fitness < local_best_fitness) {
                    local_best_fitness = fitness;
                    memcpy(local_best_pos, opt->population[i].position, opt->dim * sizeof(double));
                }
            }
            #pragma omp critical
            {
                if (local_best_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = local_best_fitness;
                    memcpy(opt->best_solution.position, local_best_pos, opt->dim * sizeof(double));
                }
            }
        }

        // Compute escaping energy
        double E1 = ENERGY_FACTOR * (1.0 - ((double)t / opt->max_iter));
        double E0 = 2.0 * hho_rand_double(0.0, 1.0, &rng_state) - 1.0;
        double Escaping_Energy = E1 * E0;

        // Exploration or exploitation
        if (fabs(Escaping_Energy) >= 1.0) {
            exploration_phase(opt, objective_function);
        } else {
            exploitation_phase(opt, objective_function);
        }
    }
}
