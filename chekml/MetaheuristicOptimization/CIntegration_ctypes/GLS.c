#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "generaloptimizer.h"
#include "GLS.h"

// Function to compute the augmented objective (g(s) + λ * sum(l_i(s) * c_i))
double evaluate_augmented_objective(Optimizer* opt, ObjectiveFunction objective_function, double* solution, double* penalties, double lambda, int dim) {
    double g = objective_function(solution);
    double penalty_term = 0.0;

    // Compute the penalty term
    for (int i = 0; i < dim; i++) {
        penalty_term += penalties[i] * solution[i]; // Use feature_indicator values here (binary indicators)
    }

    return g + lambda * penalty_term;
}

// Function for GLS optimization
void GLS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    // Ensure we have valid pointers
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Invalid optimizer or objective function.\n");
        return;
    }

    int dim = opt->dim;
    int population_size = opt->population_size;
    int max_iter = opt->max_iter;

    double lambda = 0.1;  // Penalty scaling factor (can be adjusted)
    double* penalties = (double*)calloc(dim, sizeof(double));  // Penalties for each feature
    if (!penalties) {
        fprintf(stderr, "Memory allocation failed for penalties\n");
        return;
    }

    double* current_position = (double*)malloc(dim * sizeof(double));
    double current_fitness = DBL_MAX;
    double best_fitness = DBL_MAX;

    if (!current_position) {
        fprintf(stderr, "Memory allocation failed for current_position\n");
        free(penalties);
        return;
    }

    // Initialize the best solution
    double* best_solution = (double*)malloc(dim * sizeof(double));
    if (!best_solution) {
        fprintf(stderr, "Memory allocation failed for best_solution\n");
        free(penalties);
        free(current_position);
        return;
    }

    // Random initial solution
    for (int i = 0; i < dim; i++) {
        current_position[i] = opt->bounds[2 * i] + ((double)rand() / RAND_MAX) * (opt->bounds[2 * i + 1] - opt->bounds[2 * i]);
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate the augmented objective
        current_fitness = evaluate_augmented_objective(opt, objective_function, current_position, penalties, lambda, dim);

        // Update the best solution if we find a new better fitness
        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            memcpy(best_solution, current_position, dim * sizeof(double));
        }

        // Perform local search (perturbation)
        for (int i = 0; i < dim; i++) {
            current_position[i] += (rand() / (double)RAND_MAX) * 0.2 - 0.1;  // Perturb by a small random value
            if (current_position[i] < opt->bounds[2 * i]) current_position[i] = opt->bounds[2 * i];
            if (current_position[i] > opt->bounds[2 * i + 1]) current_position[i] = opt->bounds[2 * i + 1];
        }

        // Feature indicators and penalty updates (simplified version)
        // Here we assume that feature indicators are a binary array (l_i(s)) and we are updating penalties
        double* feature_indicators = (double*)malloc(dim * sizeof(double));  // Binary indicators
        if (!feature_indicators) {
            fprintf(stderr, "Memory allocation failed for feature_indicators\n");
            free(penalties);
            free(current_position);
            free(best_solution);
            return;
        }

        // Populate feature indicators (example logic for binary indicators)
        for (int i = 0; i < dim; i++) {
            feature_indicators[i] = (current_position[i] > 0.5) ? 1.0 : 0.0;  // This is an example, modify as needed
        }

        // Update penalties
        for (int i = 0; i < dim; i++) {
            double utility = feature_indicators[i] * opt->bounds[2 * i] / (1 + penalties[i]);  // Example utility calculation
            if (utility > 0.5) penalties[i] += 1.0;
        }

        free(feature_indicators);

        // Save the history of the best solution
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, best_solution, dim * sizeof(double));

        // Print progress
        printf("Iteration: %d | Best Fitness: %lf\n", iter + 1, best_fitness);
    }

    // Clean up
    free(penalties);
    free(current_position);
    free(best_solution);
}
