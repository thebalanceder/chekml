#include "OSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize population randomly within bounds
void initialize_population_osa(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY; // Initialize fitness
    }
    enforce_bound_constraints(opt);
}

// Exploration Phase (Random Movement)
void osa_exploration_phase(Optimizer *opt, int index) {
    for (int j = 0; j < opt->dim; j++) {
        double random_move = OSA_STEP_SIZE * (2.0 * rand_double(0.0, 1.0) - 1.0);
        opt->population[index].position[j] += random_move;
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase (Move towards best solution)
void osa_exploitation_phase(Optimizer *opt, int index) {
    for (int j = 0; j < opt->dim; j++) {
        double direction = opt->best_solution.position[j] - opt->population[index].position[j];
        opt->population[index].position[j] += OSA_STEP_SIZE * direction;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void OSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize population
    initialize_population_osa(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update each individual's position
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double(0.0, 1.0) < OSA_P_EXPLORE) {
                // Exploration: Random movement
                osa_exploration_phase(opt, i);
            } else {
                // Exploitation: Move towards best solution
                osa_exploitation_phase(opt, i);
            }
        }

        // Log iteration progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
