/* BSO.c - High-Performance Implementation of Buffalo Swarm Optimization */
#include "BSO.h"
#include "generaloptimizer.h"
#include <string.h>  // For memcpy

// Xorshift PRNG for fast random number generation
static inline uint64_t bso_xorshift_next(BSOXorshiftState *rng) {
    uint64_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng->state = x;
    return x;
}

static inline double bso_xorshift_double(BSOXorshiftState *rng, double min, double max) {
    uint64_t x = bso_xorshift_next(rng);
    return min + (max - min) * ((double)x / (double)UINT64_MAX);
}

// Initialize Population
void bso_initialize_population(Optimizer *opt, BSOXorshiftState *rng) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *bounds = opt->bounds;

    // Initialize RNG state
    rng->state = (uint64_t)time(NULL) ^ (uint64_t)opt;

#if BSO_SIMD_WIDTH > 1
    // SIMD loop
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int j = 0;
        for (; j <= dim - BSO_SIMD_WIDTH; j += BSO_SIMD_WIDTH) {
            BSO_VEC_TYPE lower = BSO_VEC_SET1(bounds[2 * j]);
            BSO_VEC_TYPE upper = BSO_VEC_SET1(bounds[2 * j + 1]);
            BSO_VEC_TYPE range = BSO_VEC_ADD(upper, BSO_VEC_MUL(lower, BSO_VEC_SET1(-1.0)));
            BSO_VEC_TYPE rand_vals = BSO_VEC_SET1(bso_xorshift_double(rng, 0.0, 1.0));
            BSO_VEC_TYPE result = BSO_VEC_ADD(lower, BSO_VEC_MUL(range, rand_vals));
            BSO_VEC_STORE(&pos[j], result);
        }
        // Scalar loop for remaining dimensions
        for (; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = bso_xorshift_double(rng, lower, upper);
        }
        opt->population[i].fitness = INFINITY;
    }
#else
    // Scalar loop
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = bso_xorshift_double(rng, lower, upper);
        }
        opt->population[i].fitness = INFINITY;
    }
#endif
}

// Local Search with Integrated Bounds Enforcement
void bso_local_search(Optimizer *opt, BSOXorshiftState *rng) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *bounds = opt->bounds;
    BSO_VEC_TYPE perturb_scale = BSO_VEC_SET1(BSO_PERTURBATION_SCALE);

#if BSO_SIMD_WIDTH > 1
    // SIMD loop
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int j = 0;
        for (; j <= dim - BSO_SIMD_WIDTH; j += BSO_SIMD_WIDTH) {
            BSO_VEC_TYPE current = BSO_VEC_LOAD(&pos[j]);
            BSO_VEC_TYPE lower = BSO_VEC_SET1(bounds[2 * j]);
            BSO_VEC_TYPE upper = BSO_VEC_SET1(bounds[2 * j + 1]);
            BSO_VEC_TYPE perturb = BSO_VEC_SET1(bso_xorshift_double(rng, -1.0, 1.0));
            perturb = BSO_VEC_MUL(perturb, perturb_scale);
            BSO_VEC_TYPE new_pos = BSO_VEC_ADD(current, perturb);
            new_pos = BSO_VEC_MAX(lower, BSO_VEC_MIN(upper, new_pos));
            BSO_VEC_STORE(&pos[j], new_pos);
        }
        // Scalar loop for remaining dimensions
        for (; j < dim; j++) {
            double perturb = BSO_PERTURBATION_SCALE * bso_xorshift_double(rng, -1.0, 1.0);
            pos[j] += perturb;
            pos[j] = BSO_CLIP(pos[j], bounds[2 * j], bounds[2 * j + 1]);
        }
    }
#else
    // Scalar loop
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double perturb = BSO_PERTURBATION_SCALE * bso_xorshift_double(rng, -1.0, 1.0);
            pos[j] += perturb;
            pos[j] = BSO_CLIP(pos[j], bounds[2 * j], bounds[2 * j + 1]);
        }
    }
#endif
}

// Main Optimization Function
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize RNG
    BSOXorshiftState rng;
    rng.state = (uint64_t)time(NULL) ^ (uint64_t)opt;

    // Cache variables
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_pos = opt->best_solution.position;
    double *best_fitness = &opt->best_solution.fitness;

    // Initialize population
    bso_initialize_population(opt, &rng);

    // Set initial best solution
    *best_fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double fitness = objective_function(pos);
        opt->population[i].fitness = fitness;
        if (fitness < *best_fitness) {
            *best_fitness = fitness;
            memcpy(best_pos, pos, dim * sizeof(double));
        }
    }

    // Main optimization loop with partial unrolling
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Perform local search
        bso_local_search(opt, &rng);

        // Update fitness and best solution
        for (int i = 0; i < pop_size; i += 2) { // Unroll by 2
            // First iteration
            double *pos0 = opt->population[i].position;
            double fitness0 = objective_function(pos0);
            opt->population[i].fitness = fitness0;
            if (fitness0 < *best_fitness) {
                *best_fitness = fitness0;
                memcpy(best_pos, pos0, dim * sizeof(double));
            }

            // Second iteration (if within bounds)
            if (i + 1 < pop_size) {
                double *pos1 = opt->population[i + 1].position;
                double fitness1 = objective_function(pos1);
                opt->population[i + 1].fitness = fitness1;
                if (fitness1 < *best_fitness) {
                    *best_fitness = fitness1;
                    memcpy(best_pos, pos1, dim * sizeof(double));
                }
            }
        }
    }
}
