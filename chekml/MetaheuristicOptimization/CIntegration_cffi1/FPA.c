#include "FPA.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

// Fast Xorshift RNG for extreme speed
static uint64_t rng_state = 0x123456789ABCDEF0ULL;
static inline double fpa_rand_double(double min, double max) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t tmp = rng_state * 0x2545F4914F6CDD1DULL;
    return min + (max - min) * ((double)(tmp >> 32) / (double)0xFFFFFFFFULL);
}

// Fast normal distribution approximation (sum of uniforms)
static inline double fpa_rand_normal() {
    double sum = 0.0;
    for (int i = 0; i < 12; i++) {
        sum += fpa_rand_double(0.0, 1.0);
    }
    return sum - 6.0;
}

// Precomputed Lévy step lookup table
static double levy_lut[FPA_LEVY_LUT_SIZE];
static void init_levy_lut() {
    static int initialized = 0;
    if (initialized) return;
    const double sigma = FPA_LEVY_SIGMA;
    const double inv_beta = FPA_INV_LEVY_BETA;
    for (int i = 0; i < FPA_LEVY_LUT_SIZE; i++) {
        double v = fpa_rand_normal();
        levy_lut[i] = FPA_LEVY_STEP_SCALE * sigma / pow(fabs(v), inv_beta);
    }
    initialized = 1;
}

// Optimized Lévy flight using lookup table
static inline void fpa_levy_flight(double *restrict step, int dim) {
    for (int i = 0; i < dim; i++) {
        int lut_idx = (int)(fpa_rand_double(0.0, 1.0) * FPA_LEVY_LUT_SIZE) % FPA_LEVY_LUT_SIZE;
        step[i] = levy_lut[lut_idx];
    }
}

// Initialize Flowers (Population)
void fpa_initialize_flowers(Optimizer *restrict opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        const double *restrict bounds = opt->bounds;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = bounds[2 * j] + fpa_rand_double(0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Global Pollination Phase (SIMD-optimized)
void fpa_global_pollination_phase(Optimizer *restrict opt, double *restrict step_buffer) {
    const int vec_size = 4;  // SSE processes 4 doubles
    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) > FPA_SWITCH_PROB) {
            fpa_levy_flight(step_buffer, opt->dim);
            double *restrict pos = opt->population[i].position;
            const double *restrict best_pos = opt->best_solution.position;

            // SIMD loop for multiples of 4
            int j = 0;
            for (; j <= opt->dim - vec_size; j += vec_size) {
                __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
                __m256d step_vec = _mm256_loadu_pd(&step_buffer[j]);
                __m256d diff = _mm256_sub_pd(pos_vec, best_vec);
                __m256d update = _mm256_mul_pd(step_vec, diff);
                pos_vec = _mm256_add_pd(pos_vec, update);
                _mm256_storeu_pd(&pos[j], pos_vec);
            }
            // Handle remaining elements
            for (; j < opt->dim; j++) {
                pos[j] += step_buffer[j] * (pos[j] - best_pos[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Local Pollination Phase (SIMD-optimized)
void fpa_local_pollination_phase(Optimizer *restrict opt) {
    const int vec_size = 4;
    for (int i = 0; i < opt->population_size; i++) {
        if (fpa_rand_double(0.0, 1.0) <= FPA_SWITCH_PROB) {
            int j_idx = (int)(fpa_rand_double(0.0, 1.0) * opt->population_size);
            int k_idx = (int)(fpa_rand_double(0.0, 1.0) * opt->population_size);
            while (j_idx == k_idx) {
                k_idx = (int)(fpa_rand_double(0.0, 1.0) * opt->population_size);
            }
            double epsilon = fpa_rand_double(0.0, 1.0);
            double *restrict pos = opt->population[i].position;
            const double *restrict pos_j = opt->population[j_idx].position;
            const double *restrict pos_k = opt->population[k_idx].position;

            // SIMD loop
            __m256d eps_vec = _mm256_set1_pd(epsilon);
            int j = 0;
            for (; j <= opt->dim - vec_size; j += vec_size) {
                __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
                __m256d pos_j_vec = _mm256_loadu_pd(&pos_j[j]);
                __m256d pos_k_vec = _mm256_loadu_pd(&pos_k[j]);
                __m256d diff = _mm256_sub_pd(pos_j_vec, pos_k_vec);
                __m256d update = _mm256_mul_pd(eps_vec, diff);
                pos_vec = _mm256_add_pd(pos_vec, update);
                _mm256_storeu_pd(&pos[j], pos_vec);
            }
            // Handle remaining elements
            for (; j < opt->dim; j++) {
                pos[j] += epsilon * (pos_j[j] - pos_k[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void FPA_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict)) {
    // Stack-based step buffer (assumes dim <= FPA_MAX_DIM)
    double step_buffer[FPA_MAX_DIM] __attribute__((aligned(32)));
    if (opt->dim > FPA_MAX_DIM) {
        fprintf(stderr, "Dimension exceeds FPA_MAX_DIM\n");
        return;
    }

    // Initialize Lévy lookup table
    init_levy_lut();

    // Initialize population
    fpa_initialize_flowers(opt);

    // Evaluate initial population
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        fpa_global_pollination_phase(opt, step_buffer);
        fpa_local_pollination_phase(opt);

        // Evaluate population and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);

        // Display progress every 100 iterations
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Value = %f\n", iter, opt->best_solution.fitness);
        }
    }

    printf("Total number of evaluations: %d\n", opt->max_iter * opt->population_size);
    printf("Best solution: [");
    for (int j = 0; j < opt->dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < opt->dim - 1) printf(", ");
    }
    printf("]\n");
    printf("Best value: %f\n", opt->best_solution.fitness);
}
