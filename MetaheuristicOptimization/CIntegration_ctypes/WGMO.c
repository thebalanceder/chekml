/* WGMO.c - Implementation file for Wild Geese Migration Optimization */
#include "WGMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize geese positions randomly within bounds
void initialize_geese(Optimizer *opt) {
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer struct in initialize_geese\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

// Evaluate fitness and sort geese by fitness using indices
void evaluate_and_sort_geese(Optimizer *opt, double (*objective_function)(double *), int *indices) {
    if (!opt || !opt->population || !objective_function || !indices) {
        fprintf(stderr, "Error: Invalid inputs in evaluate_and_sort_geese\n");
        return;
    }

    // Evaluate fitness for each goose
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        indices[i] = i;  // Initialize indices
    }

    // Bubble sort indices based on fitness (ascending)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[indices[j]].fitness > opt->population[indices[j + 1]].fitness) {
                // Swap indices
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // Update best solution if necessary
    int best_idx = indices[0];
    if (opt->population[best_idx].position && opt->population[best_idx].fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = opt->population[best_idx].fitness;
        if (!opt->best_solution.position) {
            fprintf(stderr, "Error: best_solution.position is NULL\n");
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[best_idx].position[j];
        }
    }
}

// Update geese positions based on the best goose
void update_geese_positions(Optimizer *opt, int *indices) {
    if (!opt || !opt->population || !indices || !opt->population[indices[0]].position) {
        fprintf(stderr, "Error: Invalid inputs in update_geese_positions\n");
        return;
    }

    int best_idx = indices[0];  // Best goose index after sorting
    Solution best_goose = opt->population[best_idx];

    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        for (int j = 0; j < opt->dim; j++) {
            double rand_beta = rand_double(0.0, 1.0);
            double rand_gamma = rand_double(0.0, 1.0);
            double bounds_diff = opt->bounds[2 * j + 1] - opt->bounds[2 * j];

            // Update position
            opt->population[i].position[j] = (WGMO_ALPHA * opt->population[i].position[j] +
                                             WGMO_BETA * rand_beta * (best_goose.position[j] - opt->population[i].position[j]) +
                                             WGMO_GAMMA * rand_gamma * bounds_diff);

            // Enforce bounds
            opt->population[i].position[j] = fmax(opt->bounds[2 * j], 
                                                 fmin(opt->bounds[2 * j + 1], 
                                                      opt->population[i].position[j]));
        }
    }
}

// Main Optimization Function
void WGMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Invalid Optimizer or objective function\n");
        return;
    }

    // Allocate indices array for sorting
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Error: Failed to allocate indices array\n");
        return;
    }

    // Initialize geese
    initialize_geese(opt);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and sort geese indices
        evaluate_and_sort_geese(opt, objective_function, indices);

        // Update geese positions
        update_geese_positions(opt, indices);

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
