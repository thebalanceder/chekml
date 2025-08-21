#include "LS.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator

// 🔧 Function to generate a random double between min and max
double rand_double(double min, double max);

// 🌟 Generate a random solution within bounds
inline void generate_random_solution(Optimizer *opt, double *restrict solution) {
    for (int j = 0; j < opt->dim; j++) {
        solution[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
    }
}

// 🌟 Generate neighboring solutions by adding perturbations
inline void generate_neighbors(Optimizer *opt, const double *restrict current_solution, double *restrict neighbors) {
    for (int i = 0; i < NEIGHBOR_COUNT; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double perturbation = rand_double(-STEP_SIZE, STEP_SIZE);
            neighbors[i * opt->dim + j] = current_solution[j] + perturbation;
        }
    }
    // Defer bounds clipping to enforce_bound_constraints
}

// 🚀 Main Local Search Optimization Function
void LS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *restrict current_solution = (double *)malloc(opt->dim * sizeof(double));
    double *restrict best_neighbor = (double *)malloc(opt->dim * sizeof(double));
    double current_value, best_neighbor_value;
    int iteration;

    // Allocate contiguous memory for neighbors
    double *restrict neighbors = (double *)malloc(NEIGHBOR_COUNT * opt->dim * sizeof(double));
    if (!current_solution || !best_neighbor || !neighbors) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize random solution
    generate_random_solution(opt, current_solution);
    current_value = objective_function(current_solution);

    // Set initial best solution
    opt->best_solution.fitness = current_value;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = current_solution[j];
    }

    // Optimization loop
    for (iteration = 0; iteration < MAX_ITER; iteration++) {
        // Generate neighbors
        generate_neighbors(opt, current_solution, neighbors);

        // Evaluate neighbors and find the best
        best_neighbor_value = current_value;
        int best_neighbor_idx = -1;  // Track index instead of copying

        for (int i = 0; i < NEIGHBOR_COUNT; i++) {
            double neighbor_value = objective_function(&neighbors[i * opt->dim]);
            if (neighbor_value < best_neighbor_value) {
                best_neighbor_value = neighbor_value;
                best_neighbor_idx = i;
            }
        }

        // Stop if no better neighbor found
        if (best_neighbor_idx == -1) {
            printf("Stopping at iteration %d: No better neighbor found.\n", iteration + 1);
            break;
        }

        // Move to best neighbor
        for (int j = 0; j < opt->dim; j++) {
            current_solution[j] = neighbors[best_neighbor_idx * opt->dim + j];
        }
        current_value = best_neighbor_value;

        // Update global best if better
        if (best_neighbor_value < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_neighbor_value;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = current_solution[j];
            }
            // Reduce printf frequency
            if ((iteration + 1) % 10 == 0 || iteration == 0) {
                printf("Iteration %d: Best Value = %f\n", iteration + 1, opt->best_solution.fitness);
            }
        }
    }

    // Free allocated memory
    free(neighbors);
    free(current_solution);
    free(best_neighbor);

    // Ensure bounds are respected
    enforce_bound_constraints(opt);
}
