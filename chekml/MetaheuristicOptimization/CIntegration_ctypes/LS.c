#include "LS.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Generate a random solution within bounds
void generate_random_solution(Optimizer *opt, double *solution) {
    for (int j = 0; j < opt->dim; j++) {
        solution[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
    }
}

// Generate neighboring solutions by adding perturbations
void generate_neighbors(Optimizer *opt, double *current_solution, double **neighbors) {
    for (int i = 0; i < NEIGHBOR_COUNT; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double perturbation = rand_double(-STEP_SIZE, STEP_SIZE);
            neighbors[i][j] = current_solution[j] + perturbation;
            // Clip to bounds
            if (neighbors[i][j] < opt->bounds[2 * j]) {
                neighbors[i][j] = opt->bounds[2 * j];
            } else if (neighbors[i][j] > opt->bounds[2 * j + 1]) {
                neighbors[i][j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Main Local Search Optimization Function
void LS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *current_solution = (double *)malloc(opt->dim * sizeof(double));
    double *best_neighbor = (double *)malloc(opt->dim * sizeof(double));
    double current_value, best_neighbor_value;
    int iteration;

    // Initialize random solution
    generate_random_solution(opt, current_solution);
    current_value = objective_function(current_solution);

    // Set initial best solution
    opt->best_solution.fitness = current_value;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = current_solution[j];
    }

    // Allocate memory for neighbors
    double **neighbors = (double **)malloc(NEIGHBOR_COUNT * sizeof(double *));
    for (int i = 0; i < NEIGHBOR_COUNT; i++) {
        neighbors[i] = (double *)malloc(opt->dim * sizeof(double));
    }

    // Optimization loop
    for (iteration = 0; iteration < MAX_ITER; iteration++) {
        // Generate neighbors
        generate_neighbors(opt, current_solution, neighbors);

        // Evaluate neighbors and find the best
        best_neighbor_value = current_value;
        for (int j = 0; j < opt->dim; j++) {
            best_neighbor[j] = current_solution[j];
        }

        for (int i = 0; i < NEIGHBOR_COUNT; i++) {
            double neighbor_value = objective_function(neighbors[i]);
            if (neighbor_value < best_neighbor_value) {
                best_neighbor_value = neighbor_value;
                for (int j = 0; j < opt->dim; j++) {
                    best_neighbor[j] = neighbors[i][j];
                }
            }
        }

        // Stop if no better neighbor found
        if (best_neighbor_value >= current_value) {
            printf("Stopping at iteration %d: No better neighbor found.\n", iteration + 1);
            break;
        }

        // Move to best neighbor
        for (int j = 0; j < opt->dim; j++) {
            current_solution[j] = best_neighbor[j];
        }
        current_value = best_neighbor_value;

        // Update global best if better
        if (best_neighbor_value < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_neighbor_value;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = best_neighbor[j];
            }
            printf("Iteration %d: Best Value = %f\n", iteration + 1, opt->best_solution.fitness);
        }
    }

    // Free allocated memory
    for (int i = 0; i < NEIGHBOR_COUNT; i++) {
        free(neighbors[i]);
    }
    free(neighbors);
    free(current_solution);
    free(best_neighbor);

    // Ensure bounds are respected
    enforce_bound_constraints(opt);
}
