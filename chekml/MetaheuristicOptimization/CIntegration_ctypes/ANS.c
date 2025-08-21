#include "ANS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Random double in [min, max]
static double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Clamp value within bounds
static double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// ANS optimizer following ANS.m logic
void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int num_vars = opt->dim;
    int num_neighborhoods = ANS_NUM_NEIGHBORHOODS;
    int max_iterations = opt->max_iter;
    double mutation_rate = ANS_MUTATION_RATE;

    // Allocate neighborhoods
    double** populations = (double**)malloc(num_neighborhoods * sizeof(double*));
    double* fitness = (double*)malloc(num_neighborhoods * sizeof(double));
    for (int i = 0; i < num_neighborhoods; i++) {
        populations[i] = (double*)malloc(num_vars * sizeof(double));
    }

    // Initialization: Random solutions for each neighborhood
    for (int i = 0; i < num_neighborhoods; i++) {
        for (int d = 0; d < num_vars; d++) {
            double lower = opt->bounds[2 * d];
            double upper = opt->bounds[2 * d + 1];
            populations[i][d] = rand_uniform(lower, upper);
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Evaluate fitness
        for (int i = 0; i < num_neighborhoods; i++) {
            fitness[i] = objective_function(populations[i]);
        }

        // Move each neighborhood toward a random other neighborhood
        for (int i = 0; i < num_neighborhoods; i++) {
            int neighbor_index = rand() % (num_neighborhoods - 1);
            if (neighbor_index >= i) neighbor_index++;

            for (int d = 0; d < num_vars; d++) {
                double direction = populations[neighbor_index][d] - populations[i][d];
                populations[i][d] += mutation_rate * direction;

                // Clamp within bounds
                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                populations[i][d] = clamp(populations[i][d], lower, upper);
            }
        }

        // Optional: Print progress
        double best_fit = fitness[0];
        for (int i = 1; i < num_neighborhoods; i++) {
            if (fitness[i] < best_fit) {
                best_fit = fitness[i];
            }
        }
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, best_fit);
    }

    // Final best solution
    int best_idx = 0;
    for (int i = 1; i < num_neighborhoods; i++) {
        if (fitness[i] < fitness[best_idx]) {
            best_idx = i;
        }
    }

    memcpy(opt->best_solution.position, populations[best_idx], num_vars * sizeof(double));
    opt->best_solution.fitness = fitness[best_idx];

    // Clean up
    for (int i = 0; i < num_neighborhoods; i++) {
        free(populations[i]);
    }
    free(populations);
    free(fitness);
}
