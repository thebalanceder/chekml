#include "SEO.h"
#include "generaloptimizer.h"
#include <time.h>
#include <string.h>  // For memcpy

// Xorshift RNG for fast uniform random numbers
static inline unsigned long xorshift(SEO_RNG *rng) {
    unsigned long x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

// Generate uniform random number in [min, max]
static inline double seo_rng_uniform(SEO_RNG *rng, double min, double max) {
    return min + (max - min) * ((double)xorshift(rng) / (double)0xFFFFFFFFFFFFFFFF);
}

// Generate normal random number (precomputed)
static inline double seo_rng_normal(SEO_RNG *rng) {
    if (rng->normal_index >= SEO_RAND_BUFFER_SIZE) {
        // Refill normal buffer
        for (int i = 0; i < SEO_RAND_BUFFER_SIZE; i += 2) {
            double u1 = (double)xorshift(rng) / (double)0xFFFFFFFFFFFFFFFF;
            double u2 = (double)xorshift(rng) / (double)0xFFFFFFFFFFFFFFFF;
            double r = sqrt(-2.0 * log(u1));
            double theta = 2.0 * M_PI * u2;
            rng->normal_buffer[i] = r * cos(theta);
            if (i + 1 < SEO_RAND_BUFFER_SIZE) {
                rng->normal_buffer[i + 1] = r * sin(theta);
            }
        }
        rng->normal_index = 0;
    }
    return rng->normal_buffer[rng->normal_index++];
}

// Initialize RNG and precompute normal random numbers using Box-Muller
void seo_rng_init(SEO_RNG *rng, unsigned long seed) {
    rng->state = seed ? seed : (unsigned long)time(NULL);
    rng->normal_buffer = (double *)malloc(SEO_RAND_BUFFER_SIZE * sizeof(double));
    rng->normal_index = SEO_RAND_BUFFER_SIZE;  // Trigger refill on first use

    // Precompute normal random numbers
    for (int i = 0; i < SEO_RAND_BUFFER_SIZE; i += 2) {
        double u1 = seo_rng_uniform(rng, 0.0, 1.0);
        double u2 = seo_rng_uniform(rng, 0.0, 1.0);
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        rng->normal_buffer[i] = r * cos(theta);
        if (i + 1 < SEO_RAND_BUFFER_SIZE) {
            rng->normal_buffer[i + 1] = r * sin(theta);
        }
    }
    rng->normal_index = 0;
}

// Free RNG resources
void seo_rng_free(SEO_RNG *rng) {
    free(rng->normal_buffer);
    rng->normal_buffer = NULL;
}

// Initialize population randomly within bounds
static void initialize_population(Optimizer *opt, SEO_RNG *rng) {
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + seo_rng_uniform(rng, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all agents
static void evaluate_population(Optimizer *opt, double (*objective_function)(double *restrict)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Social Engineering Update Phase
void social_engineering_update(Optimizer *opt, SEO_RNG *rng) {
    for (int i = 0; i < opt->population_size; i++) {
        // Select a random target agent
        int target_index = (int)(seo_rng_uniform(rng, 0.0, opt->population_size));
        while (target_index == i) {
            target_index = (int)(seo_rng_uniform(rng, 0.0, opt->population_size));
        }

        double *restrict current = opt->population[i].position;
        double *restrict target = opt->population[target_index].position;
        double *restrict bounds_lower = opt->bounds;
        double *restrict bounds_upper = opt->bounds + 1;

        // Update position with social engineering formula
        for (int j = 0; j < opt->dim; j += 2) {
            double diff0 = target[j] - current[j];
            double diff1 = (j + 1 < opt->dim) ? target[j + 1] - current[j + 1] : 0.0;
            double randn0 = seo_rng_normal(rng);
            double randn1 = (j + 1 < opt->dim) ? seo_rng_normal(rng) : 0.0;

            current[j] += randn0 * diff0;
            if (j + 1 < opt->dim) {
                current[j + 1] += randn1 * diff1;
            }

            // Inline bounds checking
            current[j] = fmax(bounds_lower[2 * j], fmin(bounds_upper[2 * j], current[j]));
            if (j + 1 < opt->dim) {
                current[j + 1] = fmax(bounds_lower[2 * j + 2], fmin(bounds_upper[2 * j + 2], current[j + 1]));
            }
        }
    }
    // enforce_bound_constraints(opt); // Replaced with inline bounds checking
}

// Main Optimization Function
void SEO_optimize(Optimizer *opt, double (*objective_function)(double *restrict)) {
    // Initialize RNG
    SEO_RNG rng;
    seo_rng_init(&rng, (unsigned long)time(NULL));

    // Initialize population
    initialize_population(opt, &rng);
    evaluate_population(opt, objective_function);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Perform social engineering update
        social_engineering_update(opt, &rng);

        // Evaluate fitness and update population
        int best_idx = 0;
        double best_fitness = opt->population[0].fitness;

        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;

            // Track best solution
            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                best_idx = i;
            }
        }

        // Update best solution
        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_fitness;
            memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
        }

        // Optional logging (comment out for max performance)
        // printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free RNG resources
    seo_rng_free(&rng);
}
