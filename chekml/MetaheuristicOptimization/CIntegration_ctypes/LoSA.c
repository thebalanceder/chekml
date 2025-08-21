#include "LoSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void LoSA_initialize_population(Optimizer *opt) {
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "Error: Null pointer in LoSA_initialize_population\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness to infinity
    }
    enforce_bound_constraints(opt);
}

// Update Positions
void LoSA_update_positions(Optimizer *opt) {
    if (!opt || !opt->population || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in LoSA_update_positions\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double direction = opt->best_solution.position[j] - opt->population[i].position[j];
            opt->population[i].position[j] += LOSA_STEP_SIZE * direction * rand_double(0.0, 1.0);
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds || opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid parameters in LoSA_optimize\n");
        return;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize population
    LoSA_initialize_population(opt);

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Update positions
        LoSA_update_positions(opt);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
