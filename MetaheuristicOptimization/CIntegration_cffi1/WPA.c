/* WPA.c - Implementation file for Waterwheel Plant Algorithm (WWPA) */
#include "WPA.h"
#include "generaloptimizer.h"
#include <stdint.h>  // For uint64_t
#include <time.h>    // For time()
#include <string.h>  // For memcpy

// Fast Xorshift RNG (period 2^64-1)
static inline double wpa_fast_rand(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return ((double)(*state * 0x2545F4914F6CDD1DULL) / 0xFFFFFFFFFFFFFFFFULL);
}

// Fast Gaussian approximation (simplified Ziggurat-like method)
static inline double wpa_fast_gaussian(uint64_t *state) {
    double x = wpa_fast_rand(state);
    double y = wpa_fast_rand(state);
    double z = x * 2.0 - 1.0;  // Map to [-1, 1]
    return z * (1.0 - y * y);  // Approximate normal distribution
}

// Compute population mean and standard deviation
static inline void compute_stats(Optimizer *restrict opt, double *restrict mu_P, double *restrict sigma) {
    register const int dim = opt->dim;
    register const int pop_size = opt->population_size;
    register double *pop_pos;
    register double sum, sum_sq;

    for (register int j = 0; j < dim; j++) {
        sum = 0.0;
        sum_sq = 0.0;
        for (register int i = 0; i < pop_size; i++) {
            pop_pos = opt->population[i].position;
            register double val = pop_pos[j];
            sum += val;
            sum_sq += val * val;
        }
        mu_P[j] = sum / pop_size;
        sigma[j] = sqrt(sum_sq / pop_size - mu_P[j] * mu_P[j]);
    }
}

// Exploration Phase (Position Identification and Hunting)
void wpa_exploration_phase(Optimizer *restrict opt, int t, double K, int *restrict stagnation_counts, 
                          double *restrict mu_P, double *restrict sigma, double *restrict rand_buffer, 
                          double (*objective_function)(double *)) {
    register const int dim = opt->dim;
    register const int pop_size = opt->population_size;
    double *W = rand_buffer;  // Reuse rand_buffer for W
    double *new_position = rand_buffer + dim;  // Reuse rand_buffer for new_position
    register double *pop_pos, *bounds;
    uint64_t rng_state = (uint64_t)t + 1;  // Simple seed per iteration

    for (register int i = 0; i < pop_size; i++) {
        register double r1 = R1_MIN + (R1_MAX - R1_MIN) * wpa_fast_rand(&rng_state);
        register double r2 = R2_MIN + (R2_MAX - R2_MIN) * wpa_fast_rand(&rng_state);
        pop_pos = opt->population[i].position;
        bounds = opt->bounds;

        // Equation (4): W = r1 * (P(t) + 2K)
        register double two_K = 2.0 * K;
        for (register int j = 0; j < dim; j++) {
            W[j] = r1 * (pop_pos[j] + two_K);
        }

        // Equation (5): P(t+1) = P(t) + W * (2K + r2)
        register double factor = two_K + r2;
        for (register int j = 0; j < dim; j++) {
            new_position[j] = pop_pos[j] + W[j] * factor;
            new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
        }

        // Evaluate fitness
        register double new_value = objective_function(new_position);
        register double current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            memcpy(pop_pos, new_position, dim * sizeof(double));
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
                for (register int j = 0; j < dim; j++) {
                    register double gaussian_term = mu_P[j] + sigma[j] * wpa_fast_gaussian(&rng_state);
                    new_position[j] = (W[j] != 0.0) ? gaussian_term + r1 * (pop_pos[j] + two_K) / W[j] : gaussian_term;
                    new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                memcpy(pop_pos, new_position, dim * sizeof(double));
                opt->population[i].fitness = objective_function(new_position);
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Exploitation Phase (Carrying Insect to Suitable Tube)
void wpa_exploitation_phase(Optimizer *restrict opt, int t, double K, int *restrict stagnation_counts, 
                           double *restrict rand_buffer, double (*objective_function)(double *)) {
    register const int dim = opt->dim;
    register const int pop_size = opt->population_size;
    double *W = rand_buffer;  // Reuse rand_buffer for W
    double *new_position = rand_buffer + dim;  // Reuse rand_buffer for new_position
    register double *pop_pos, *best_pos, *bounds;
    uint64_t rng_state = (uint64_t)t + 1;

    for (register int i = 0; i < pop_size; i++) {
        register double r3 = R3_MIN + (R3_MAX - R3_MIN) * wpa_fast_rand(&rng_state);
        pop_pos = opt->population[i].position;
        best_pos = opt->best_solution.position;
        bounds = opt->bounds;

        // Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
        for (register int j = 0; j < dim; j++) {
            W[j] = r3 * (K * best_pos[j] + r3 * pop_pos[j]);
        }

        // Equation (8): P(t+1) = P(t) + K * W
        for (register int j = 0; j < dim; j++) {
            new_position[j] = pop_pos[j] + K * W[j];
            new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
        }

        // Evaluate fitness
        register double new_value = objective_function(new_position);
        register double current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            memcpy(pop_pos, new_position, dim * sizeof(double));
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
                register double r1 = R1_MIN + (R1_MAX - R1_MIN) * wpa_fast_rand(&rng_state);
                register double f = F_MIN + (F_MAX - F_MIN) * wpa_fast_rand(&rng_state);
                register double c = C_MIN + (C_MAX - C_MIN) * wpa_fast_rand(&rng_state);
                register double theta = 2.0 * PI * wpa_fast_rand(&rng_state);
                register double factor = (c != 0.0) ? (r1 + K) * sin(f / c * theta) : pop_pos[0];

                for (register int j = 0; j < dim; j++) {
                    new_position[j] = factor;
                    new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                memcpy(pop_pos, new_position, dim * sizeof(double));
                opt->population[i].fitness = objective_function(new_position);
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Update K Parameter (Equation 10)
static inline double wpa_update_k(int t, int max_iter, double f) {
    return 1.0 + (2.0 * t * t / (max_iter * max_iter)) + f;
}

// Main Optimization Function
void WPA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    register const int pop_size = opt->population_size;
    register const int dim = opt->dim;
    register const int max_iter = opt->max_iter;
    int *stagnation_counts = (int *)calloc(pop_size, sizeof(int));
    double *mu_P = (double *)calloc(dim, sizeof(double));
    double *sigma = (double *)calloc(dim, sizeof(double));
    double *rand_buffer = (double *)calloc(2 * dim, sizeof(double));  // For W and new_position
    uint64_t rng_state = (uint64_t)time(NULL);

    for (register int iter = 0; iter < max_iter; iter++) {
        register double r = wpa_fast_rand(&rng_state);
        register double f = F_MIN + (F_MAX - F_MIN) * wpa_fast_rand(&rng_state);
        register double K = wpa_update_k(iter, max_iter, f);

        if (r < 0.5) {
            compute_stats(opt, mu_P, sigma);
            wpa_exploration_phase(opt, iter, K, stagnation_counts, mu_P, sigma, rand_buffer, objective_function);
        } else {
            wpa_exploitation_phase(opt, iter, K, stagnation_counts, rand_buffer, objective_function);
        }

        // Update best solution
        register double *pop_pos, *best_pos = opt->best_solution.position;
        for (register int i = 0; i < pop_size; i++) {
            pop_pos = opt->population[i].position;
            register double new_fitness = objective_function(pop_pos);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(best_pos, pop_pos, dim * sizeof(double));
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(stagnation_counts);
    free(mu_P);
    free(sigma);
    free(rand_buffer);
}
