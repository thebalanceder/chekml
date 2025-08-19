#include "POA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator

// Initialize Population
void initialize_population_poa(Optimizer *restrict opt) {
    // ✅ Initialize population randomly within bounds
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        const double *bounds = opt->bounds;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = rand_double_poa(bounds[2 * j], bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    // ✅ Enforce bounds once after initialization
    enforce_bound_constraints(opt);
}

// Evaluate Population and Update Best Solution
void evaluate_and_update_best(Optimizer *restrict opt, double (*objective_function)(double *)) {
    // ✅ Combine evaluation and best solution update to reduce loops
    double best_fitness = opt->best_solution.fitness;
    int best_idx = -1;

    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
        // Update best solution if a better one was found
        if (best_idx >= 0) {
            opt->best_solution.fitness = best_fitness;
            double *best_pos = opt->best_solution.position;
            const double *new_best_pos = opt->population[best_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = new_best_pos[j];
            }
        }
    }
}

// Update Positions Towards Best Solution
void update_positions_poa(Optimizer *restrict opt) {
    // ✅ Optimize position updates with integrated bounds checking
    const double *best_pos = opt->best_solution.position;
    const double *bounds = opt->bounds;

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double norm = 0.0;
        double direction[opt->dim];

        // Calculate direction and norm
        for (int j = 0; j < opt->dim; j++) {
            direction[j] = best_pos[j] - pos[j];
            norm += direction[j] * direction[j];
        }
        norm = sqrt(norm);

        // Update position with normalized direction and bounds checking
        if (norm > 0.0) {
            double inv_norm = STEP_SIZE / norm;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += direction[j] * inv_norm;
                // Inline bounds enforcement
                if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
                else if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
            }
        }
    }
    // ✅ No need for separate enforce_bound_constraints call
}

// Main Optimization Function
void POA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    // ✅ Initialize population
    initialize_population_poa(opt);

    // ✅ Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and update best solution
        evaluate_and_update_best(opt, objective_function);

        // Update positions towards the best solution
        update_positions_poa(opt);

        // Log iteration progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
