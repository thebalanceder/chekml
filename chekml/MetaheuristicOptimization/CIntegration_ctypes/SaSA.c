#include "SaSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize salp population
void sasa_initialize_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY; // Updated in optimize
    }
}

// Leader salp update phase (Eq. 3.1)
void sasa_leader_update(Optimizer *opt, double c1) {
    for (int i = 0; i < opt->population_size / 2; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            double c2 = rand_double(0.0, 1.0);
            double c3 = rand_double(0.0, 1.0);
            double new_position;
            if (c3 < 0.5) {
                new_position = opt->best_solution.position[j] + c1 * ((ub - lb) * c2 + lb);
            } else {
                new_position = opt->best_solution.position[j] - c1 * ((ub - lb) * c2 + lb);
            }
            opt->population[i].position[j] = new_position;
        }
    }
    enforce_bound_constraints(opt);
}

// Follower salp update phase (Eq. 3.4)
void sasa_follower_update(Optimizer *opt) {
    for (int i = opt->population_size / 2; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = (opt->population[i - 1].position[j] + 
                                             opt->population[i].position[j]) / 2.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "SaSA_optimize: Invalid optimizer or objective function\n");
        return;
    }

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Initialize population
    sasa_initialize_population(opt);

    // Evaluate initial fitness and set best solution
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Sort population by fitness (bubble sort, swapping values not pointers)
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "SaSA_optimize: Memory allocation failed for temp_position\n");
        return;
    }

    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                // Swap fitness
                double temp_fitness = opt->population[j].fitness;
                opt->population[j].fitness = opt->population[j + 1].fitness;
                opt->population[j + 1].fitness = temp_fitness;

                // Swap position values (not pointers)
                for (int k = 0; k < opt->dim; k++) {
                    temp_position[k] = opt->population[j].position[k];
                    opt->population[j].position[k] = opt->population[j + 1].position[k];
                    opt->population[j + 1].position[k] = temp_position[k];
                }
            }
        }
    }
    free(temp_position);

    // Main optimization loop
    for (int iter = 1; iter <= opt->max_iter; iter++) {
        // Calculate c1 coefficient (Eq. 3.2)
        double c1 = SASA_C1_FACTOR * exp(-pow((SASA_C1_EXPONENT * iter / opt->max_iter), 2));

        // Update leader salps
        sasa_leader_update(opt, c1);

        // Update follower salps
        sasa_follower_update(opt);

        // Evaluate fitness and update best solution
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

        // Log progress
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    // Ensure best solution is within bounds
    enforce_bound_constraints(opt);
}
