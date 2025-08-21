#include "SEO.h"
#include "generaloptimizer.h"
#include <time.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

// Cache line size (64 bytes)
#define CACHE_LINE 64

// Xorshift128+ RNG
static inline unsigned long long xorshift128p(SEO_RNG *rng) {
    unsigned long long x = rng->state[0];
    unsigned long long y = rng->state[1];
    rng->state[0] = y;
    x ^= x << 23;
    rng->state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rng->state[1] + y;
}

// Generate uniform random number in [0, 1]
static inline double seo_rng_uniform(SEO_RNG *rng) {
    return (double)(xorshift128p(rng) >> 11) * (1.0 / (1ULL << 53));
}

// Generate normal random number (Box-Muller)
static inline double seo_rng_normal(SEO_RNG *rng) {
    if (rng->normal_index >= SEO_RAND_BUFFER_SIZE) {
        for (int i = 0; i < SEO_RAND_BUFFER_SIZE; i += 2) {
            double u1 = seo_rng_uniform(rng);
            double u2 = seo_rng_uniform(rng);
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

// Initialize RNG
void seo_rng_init(SEO_RNG *rng, unsigned long seed) {
    rng->state[0] = seed;
    rng->state[1] = seed ^ 0xDEADBEEF;
    rng->normal_index = SEO_RAND_BUFFER_SIZE;
    if (posix_memalign((void **)&rng->normal_buffer, CACHE_LINE, SEO_RAND_BUFFER_SIZE * sizeof(double)) != 0) {
        rng->normal_buffer = NULL;
    }
}

// Free RNG
void seo_rng_free(SEO_RNG *rng) {
    free(rng->normal_buffer);
    rng->normal_buffer = NULL;
}

// Social Engineering Update
void social_engineering_update(Optimizer *opt, SEO_RNG *rng) {
    assert(opt != NULL && opt->population != NULL && opt->bounds != NULL);

    for (int i = 0; i < opt->population_size; i++) {
        int target_idx = (int)(seo_rng_uniform(rng) * opt->population_size);
        while (target_idx == i) {
            target_idx = (int)(seo_rng_uniform(rng) * opt->population_size);
        }

        double *restrict current = opt->population[i].position;
        double *restrict target = opt->population[target_idx].position;
        for (int j = 0; j < opt->dim; j++) {
            double diff = target[j] - current[j];
            current[j] += seo_rng_normal(rng) * diff;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SEO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    assert(opt_void != NULL && objective_function != NULL);

    Optimizer *opt = (Optimizer *)opt_void;
    assert(opt->population != NULL && opt->bounds != NULL);

    // Initialize RNG
    SEO_RNG rng;
    seo_rng_init(&rng, (unsigned long)time(NULL));
    if (rng.normal_buffer == NULL) {
        return;  // Silently exit on RNG allocation failure
    }

    // Initial fitness evaluation
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        social_engineering_update(opt, &rng);

        // Fitness evaluation and best solution update
        int best_idx = 0;
        double best_fitness = opt->population[0].fitness;

        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < best_fitness) {
                best_fitness = opt->population[i].fitness;
                best_idx = i;
            }
        }

        if (best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_fitness;
            memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
        }
    }

    seo_rng_free(&rng);
}
