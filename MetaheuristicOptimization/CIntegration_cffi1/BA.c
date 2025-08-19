#include "BA.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// 🦇 Initialize Xorshift RNG state
void init_xorshift_state(XorshiftState_Ba *restrict state, uint64_t seed) {
    state->s[0] = seed ^ 0x123456789ABCDEFULL;
    state->s[1] = seed ^ 0xFEDCBA9876543210ULL;
    // Warm up the RNG
    for (int i = 0; i < 10; i++) xorshift_double_ba(state);
}

// 🦇 Frequency Update and Velocity Adjustment Phase
void bat_frequency_update(Optimizer *restrict opt, double *restrict freq, double *restrict velocities, XorshiftState_Ba *restrict rng_states) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double freq_range = FREQ_MAX - FREQ_MIN;
    const int vec_dim = (dim + 3) & ~3; // Pad to multiple of 4 for AVX

    #pragma omp parallel
    {
        XorshiftState_Ba rng = rng_states[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int i = 0; i < pop_size; i++) {
            // Update frequency
            freq[i] = FREQ_MIN + freq_range * xorshift_double_ba(&rng);

            // Update velocity and position with AVX
            double *pos = opt->population[i].position;
            double *vel = velocities + i * vec_dim;
            const double *best_pos = opt->best_solution.position;
            const __m256d freq_vec = _mm256_set1_pd(freq[i]);

            for (int j = 0; j < dim; j += 4) {
                __m256d pos_vec = _mm256_load_pd(pos + j);
                __m256d vel_vec = _mm256_load_pd(vel + j);
                __m256d best_vec = _mm256_load_pd(best_pos + j);
                __m256d diff = _mm256_sub_pd(pos_vec, best_vec);
                vel_vec = _mm256_add_pd(vel_vec, _mm256_mul_pd(diff, freq_vec));
                pos_vec = _mm256_add_pd(pos_vec, vel_vec);
                _mm256_store_pd(vel + j, vel_vec);
                _mm256_store_pd(pos + j, pos_vec);
            }
            rng_states[omp_get_thread_num()] = rng;
        }
    }

    enforce_bound_constraints(opt);
}

// 🦇 Local Search Phase
void bat_local_search(Optimizer *restrict opt, double *restrict freq, double pulse_rate, double loudness, XorshiftState_Ba *restrict rng_states) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const int vec_dim = (dim + 3) & ~3; // Pad to multiple of 4
    const double scale = LOCAL_SEARCH_SCALE * loudness;

    #pragma omp parallel
    {
        XorshiftState_Ba rng = rng_states[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int i = 0; i < pop_size; i++) {
            if (xorshift_double_ba(&rng) < pulse_rate) {
                double *pos = opt->population[i].position;
                const double *best_pos = opt->best_solution.position;
                for (int j = 0; j < dim; j += 4) {
                    __m256d best_vec = _mm256_load_pd(best_pos + j);
                    __m256d rand_vec = _mm256_set_pd(
                        xorshift_double_ba(&rng), xorshift_double_ba(&rng),
                        xorshift_double_ba(&rng), xorshift_double_ba(&rng));
                    rand_vec = _mm256_sub_pd(_mm256_mul_pd(rand_vec, _mm256_set1_pd(2.0)), _mm256_set1_pd(1.0));
                    __m256d pos_vec = _mm256_add_pd(best_vec, _mm256_mul_pd(rand_vec, _mm256_set1_pd(scale)));
                    _mm256_store_pd(pos + j, pos_vec);
                }
            }
            rng_states[omp_get_thread_num()] = rng;
        }
    }

    enforce_bound_constraints(opt);
}

// 🦇 Solution Update Phase
void bat_update_solutions(Optimizer *restrict opt, double *restrict freq, double loudness, double (*objective_function)(double *), XorshiftState_Ba *restrict rng_states) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;

    #pragma omp parallel
    {
        XorshiftState_Ba rng = rng_states[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int i = 0; i < pop_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness <= opt->population[i].fitness && xorshift_double_ba(&rng) > loudness) {
                opt->population[i].fitness = new_fitness;
            }
            #pragma omp critical
            {
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    double *pos = opt->population[i].position;
                    double *best_pos = opt->best_solution.position;
                    for (int j = 0; j < dim; j++) {
                        best_pos[j] = pos[j];
                    }
                }
            }
            rng_states[omp_get_thread_num()] = rng;
        }
    }
}

// 🚀 Main Optimization Function
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    // Align population and velocities
    const int dim = opt->dim;
    const int vec_dim = (dim + 3) & ~3; // Pad to multiple of 4 for AVX
    const int pop_size = opt->population_size;

    // Allocate aligned memory
    double *freq = (double *)_mm_malloc(pop_size * sizeof(double), ALIGNMENT);
    double *velocities = (double *)_mm_malloc(pop_size * vec_dim * sizeof(double), ALIGNMENT);
    XorshiftState_Ba *rng_states = (XorshiftState_Ba *)malloc(omp_get_max_threads() * sizeof(XorshiftState_Ba));
    if (!freq || !velocities || !rng_states) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Zero-initialize velocities
    for (int i = 0; i < pop_size * vec_dim; i++) {
        velocities[i] = 0.0;
    }

    // Initialize RNG states
    uint64_t seed = (uint64_t)time(NULL);
    for (int i = 0; i < omp_get_max_threads(); i++) {
        init_xorshift_state(&rng_states[i], seed + i);
    }

    // Initialize population fitness
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        #pragma omp critical
        {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                double *pos = opt->population[i].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }
    }

    // Main optimization loop
    double loudness = LOUDNESS;
    double pulse_rate = PULSE_RATE;
    for (int iter = 0; iter < opt->max_iter; iter++) {
        pulse_rate = PULSE_RATE * (1.0 - exp(-GAMMA * iter));
        loudness *= ALPHA_BA;

        bat_frequency_update(opt, freq, velocities, rng_states);
        bat_local_search(opt, freq, pulse_rate, loudness, rng_states);
        bat_update_solutions(opt, freq, loudness, objective_function, rng_states);

        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }

    // Clean up
    _mm_free(freq);
    _mm_free(velocities);
    free(rng_states);
}
