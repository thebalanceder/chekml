#include "POA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void initialize_population_poa(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY; // Initialize fitness to infinity
    }
    enforce_bound_constraints(opt);
}

// Evaluate Population Fitness
void evaluate_population_poa(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Update Positions Towards Best Solution
void update_positions_poa(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double norm = 0.0;
        double direction[opt->dim];

        // Calculate direction towards the best solution
        for (int j = 0; j < opt->dim; j++) {
            direction[j] = opt->best_solution.position[j] - opt->population[i].position[j];
            norm += direction[j] * direction[j];
        }
        norm = sqrt(norm);

        // Normalize direction (handle zero norm case)
        if (norm != 0.0) {
            for (int j = 0; j < opt->dim; j++) {
                direction[j] /= norm;
            }
        }

        // Update position
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += STEP_SIZE * direction[j];
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void POA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize population
    initialize_population_poa(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        evaluate_population_poa(opt, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update positions towards the best solution
        update_positions_poa(opt);

        // Log iteration progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
