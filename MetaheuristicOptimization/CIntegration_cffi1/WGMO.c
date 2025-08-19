/* WGMO.c - Extreme Speed Implementation for Wild Geese Migration Optimization */
#include "WGMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For malloc/free
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

// Initialize geese positions randomly within bounds (unrolled for dim=2)
void initialize_geese(Optimizer *opt, WGMOXorshiftState *rng) {
    if (!opt || !opt->population || !opt->bounds || !rng) {
        fprintf(stderr, "Error: Invalid Optimizer struct in initialize_geese\n");
        return;
    }

    double *restrict bounds = opt->bounds;
    Solution *restrict pop = opt->population;
    for (int i = 0; i < opt->population_size; i++) {
        if (!pop[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            return;
        }
        double *pos = pop[i].position;
        pos[0] = wgmo_xorshift_double(rng, bounds[0], bounds[1]);  // dim=0
        pos[1] = wgmo_xorshift_double(rng, bounds[2], bounds[3]);  // dim=1
    }
}

// Evaluate fitness and sort geese by fitness using insertion sort
void evaluate_and_sort_geese(Optimizer *opt, double (*objective_function)(double *), int *indices) {
    if (!opt || !opt->population || !objective_function || !indices) {
        fprintf(stderr, "Error: Invalid inputs in evaluate_and_sort_geese\n");
        return;
    }

    // Evaluate fitness and initialize indices
    Solution *restrict pop = opt->population;
    for (int i = 0; i < opt->population_size; i++) {
        if (!pop[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            indices[i] = i;
            pop[i].fitness = INFINITY;
            continue;
        }
        pop[i].fitness = objective_function(pop[i].position);
        indices[i] = i;
    }

    // Insertion sort indices by fitness (faster for small n=50, nearly sorted)
    for (int i = 1; i < opt->population_size; i++) {
        int key = indices[i];
        double key_fitness = pop[key].fitness;
        int j = i - 1;
        while (j >= 0 && pop[indices[j]].fitness > key_fitness) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    // Update best solution if necessary
    int best_idx = indices[0];
    if (pop[best_idx].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = pop[best_idx].fitness;
        if (!opt->best_solution.position) {
            fprintf(stderr, "Error: best_solution.position is NULL\n");
            return;
        }
        double *restrict best_pos = opt->best_solution.position;
        double *restrict pop_pos = pop[best_idx].position;
        best_pos[0] = pop_pos[0];
        best_pos[1] = pop_pos[1];
    }
}

// Update geese positions based on the best goose (unrolled for dim=2)
void update_geese_positions(Optimizer *opt, int *indices, double *rand_buffer, int rand_offset) {
    if (!opt || !opt->population || !indices || !rand_buffer || !opt->population[indices[0]].position) {
        fprintf(stderr, "Error: Invalid inputs in update_geese_positions\n");
        return;
    }

    double *restrict bounds = opt->bounds;
    Solution *restrict pop = opt->population;
    int best_idx = indices[0];
    double *restrict best_pos = pop[best_idx].position;

    for (int i = 0; i < opt->population_size; i++) {
        if (!pop[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        double *restrict pos = pop[i].position;
        double r_beta = rand_buffer[rand_offset + i * 4 + 0];    // beta for dim=0
        double r_gamma = rand_buffer[rand_offset + i * 4 + 1];   // gamma for dim=0
        double r_beta2 = rand_buffer[rand_offset + i * 4 + 2];   // beta for dim=1
        double r_gamma2 = rand_buffer[rand_offset + i * 4 + 3];  // gamma for dim=1

        // Update position for dim=0
        pos[0] = (WGMO_ALPHA * pos[0] +
                  WGMO_BETA * r_beta * (best_pos[0] - pos[0]) +
                  WGMO_GAMMA * r_gamma * (bounds[1] - bounds[0]));
        pos[0] = fmax(bounds[0], fmin(bounds[1], pos[0]));

        // Update position for dim=1
        pos[1] = (WGMO_ALPHA * pos[1] +
                  WGMO_BETA * r_beta2 * (best_pos[1] - pos[1]) +
                  WGMO_GAMMA * r_gamma2 * (bounds[3] - bounds[2]));
        pos[1] = fmax(bounds[2], fmin(bounds[3], pos[1]));
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

    // Pre-generate random numbers (4 per goose: beta, gamma for each dim)
    int rand_count = opt->max_iter * opt->population_size * 4;
    double *rand_buffer = (double *)malloc(rand_count * sizeof(double));
    if (!rand_buffer) {
        fprintf(stderr, "Error: Failed to allocate rand_buffer\n");
        free(indices);
        return;
    }
    for (int i = 0; i < rand_count; i++) {
        rand_buffer[i] = wgmo_xorshift_double(&rng, 0.0, 1.0);
    }

    // Initialize geese
    initialize_geese(opt, &rng);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and sort geese indices
        evaluate_and_sort_geese(opt, objective_function, indices);

        // Update geese positions
        int rand_offset = iter * opt->population_size * 4;
        update_geese_positions(opt, indices, rand_buffer, rand_offset);

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free allocated memory
    free(rand_buffer);
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
