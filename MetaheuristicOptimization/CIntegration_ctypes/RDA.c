/* RDA.c - Implementation file for Red Deer Algorithm (RDA) Optimization */
#include "RDA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Update Position Phase (combines exploration and exploitation)
void update_position_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double r = rand_double(0.0, 1.0);

        if (r < P_EXPLORATION) {
            // Exploration: Move randomly
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] += STEP_SIZE * (rand_double(0.0, 1.0) * 2 - 1);
            }
        } else {
            // Exploitation: Move towards the best solution
            for (int j = 0; j < opt->dim; j++) {
                double direction = opt->best_solution.position[j] - opt->population[i].position[j];
                opt->population[i].position[j] += STEP_SIZE * direction;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void RDA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update positions
        update_position_phase(opt);

        // Evaluate fitness and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;  // Update fitness
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);

        // Optional: Log progress (similar to Python's print statement)
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
