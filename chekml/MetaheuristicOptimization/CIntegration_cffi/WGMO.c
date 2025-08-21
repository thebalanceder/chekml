/* WGMO.c - Optimized Implementation file for Wild Geese Migration Optimization */
#include "WGMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For malloc/free/qsort
#include <stdio.h>   // For debugging/logging
#include <time.h>    // For seeding Xorshift
#include <stdint.h>  // For uint64_t

// WGMO-specific Xorshift random number generator
static void wgmo_xorshift_init(WGMOXorshiftState *state, uint64_t seed) {
    state->state = seed ? seed : (uint64_t)time(NULL);
    if (state->state == 0) state->state = 1;  // Ensure non-zero state
}

static inline uint64_t wgmo_xorshift_next(WGMOXorshiftState *state) {
    uint64_t x = state->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state->state = x;
    return x;
}

static inline double wgmo_xorshift_double(WGMOXorshiftState *state, double min, double max) {
    uint64_t r = wgmo_xorshift_next(state);
    return min + (max - min) * ((double)r / (double)UINT64_MAX);
}

// Global context for qsort
typedef struct {
    Solution *population;
} SortContext;

static SortContext sort_ctx;

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    double fa = sort_ctx.population[idx_a].fitness;
    double fb = sort_ctx.population[idx_b].fitness;
    return (fa > fb) - (fa < fb);  // Ascending order
}

// Initialize geese positions randomly within bounds
void initialize_geese(Optimizer *opt, WGMOXorshiftState *rng) {
    if (!opt || !opt->population || !opt->bounds || !rng) {
        fprintf(stderr, "Error: Invalid Optimizer struct in initialize_geese\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            return;
        }
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = wgmo_xorshift_double(rng, opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

// Evaluate fitness and sort geese by fitness using indices
void evaluate_and_sort_geese(Optimizer *opt, double (*objective_function)(double *), int *indices) {
    if (!opt || !opt->population || !objective_function || !indices) {
        fprintf(stderr, "Error: Invalid inputs in evaluate_and_sort_geese\n");
        return;
    }

    // Evaluate fitness and initialize indices
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            indices[i] = i;
            opt->population[i].fitness = INFINITY;
            continue;
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        indices[i] = i;
    }

    // Sort indices using qsort
    sort_ctx.population = opt->population;
    qsort(indices, opt->population_size, sizeof(int), compare_fitness);

    // Update best solution if necessary
    int best_idx = indices[0];
    if (opt->population[best_idx].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[best_idx].fitness;
        if (!opt->best_solution.position) {
            fprintf(stderr, "Error: best_solution.position is NULL\n");
            return;
        }
        double *best_pos = opt->best_solution.position;
        double *pop_pos = opt->population[best_idx].position;
        for (int j = 0; j < opt->dim; j++) {
            best_pos[j] = pop_pos[j];
        }
    }
}

// Update geese positions based on the best goose
void update_geese_positions(Optimizer *opt, int *indices, WGMOXorshiftState *rng) {
    if (!opt || !opt->population || !indices || !rng || !opt->population[indices[0]].position) {
        fprintf(stderr, "Error: Invalid inputs in update_geese_positions\n");
        return;
    }

    int best_idx = indices[0];
    double *best_pos = opt->population[best_idx].position;

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double r_beta = wgmo_xorshift_double(rng, 0.0, 1.0);
            double r_gamma = wgmo_xorshift_double(rng, 0.0, 1.0);
            double bounds_diff = opt->bounds[2 * j + 1] - opt->bounds[2 * j];

            // Update position
            pos[j] = (WGMO_ALPHA * pos[j] +
                      WGMO_BETA * r_beta * (best_pos[j] - pos[j]) +
                      WGMO_GAMMA * r_gamma * bounds_diff);

            // Enforce bounds
            pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], pos[j]));
        }
    }
}

// Main Optimization Function
void WGMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer or objective function\n");
        return;
    }

    // Initialize WGMO-specific Xorshift RNG
    WGMOXorshiftState rng;
    wgmo_xorshift_init(&rng, (uint64_t)time(NULL));

    // Allocate indices array for sorting
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Error: Failed to allocate indices array\n");
        return;
    }

    // Initialize geese
    initialize_geese(opt, &rng);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and sort geese indices
        evaluate_and_sort_geese(opt, objective_function, indices);

        // Update geese positions
        update_geese_positions(opt, indices, &rng);

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free indices array
    free(indices);

    // Final output
    printf("\nOptimization finished.\n");
    printf("Best solution found: [");
    if (opt->best_solution.position) {
        for (int j = 0; j < opt->dim; j++) {
            printf("%f", opt->best_solution.position[j]);
            if (j < opt->dim - 1) printf(", ");
        }
    } else {
        printf("NULL");
    }
    printf("]\n");
    printf("Best fitness: %f\n", opt->best_solution.fitness);
}
