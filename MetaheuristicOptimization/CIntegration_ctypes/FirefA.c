#include "FirefA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize firefly population
void initialize_fireflies(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Update firefly positions based on attractiveness and randomness
void update_fireflies(Optimizer *opt, int t, ObjectiveFunction objective_function) {
    double alpha = FA_ALPHA * pow(FA_THETA, t);  // Reduce randomness over time
    double scale[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        scale[j] = fabs(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
    }

    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        for (int j = 0; j < opt->population_size; j++) {
            if (opt->population[i].fitness >= opt->population[j].fitness) {  // Move if j is brighter
                double r = 0.0;
                for (int k = 0; k < opt->dim; k++) {
                    r += pow(opt->population[i].position[k] - opt->population[j].position[k], 2);
                }
                r = sqrt(r);
                double beta = FA_BETA0 * exp(-FA_GAMMA * r * r);
                for (int k = 0; k < opt->dim; k++) {
                    double step = alpha * (rand_double(0.0, 1.0) - 0.5) * scale[k];
                    opt->population[i].position[k] += beta * (opt->population[j].position[k] - opt->population[i].position[k]) + step;
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Rank fireflies by fitness and sort population
void rank_fireflies(Optimizer *opt) {
    // Simple bubble sort for ranking (could be optimized with qsort)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                // Swap solutions
                double temp_fitness = opt->population[j].fitness;
                double temp_position[opt->dim];
                for (int k = 0; k < opt->dim; k++) {
                    temp_position[k] = opt->population[j].position[k];
                }
                opt->population[j].fitness = opt->population[j + 1].fitness;
                for (int k = 0; k < opt->dim; k++) {
                    opt->population[j].position[k] = opt->population[j + 1].position[k];
                }
                opt->population[j + 1].fitness = temp_fitness;
                for (int k = 0; k < opt->dim; k++) {
                    opt->population[j + 1].position[k] = temp_position[k];
                }
            }
        }
    }
}

// Main Optimization Function
void FirefA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    initialize_fireflies(opt, objective_function);

    for (int t = 0; t < opt->max_iter; t++) {
        update_fireflies(opt, t, objective_function);
        rank_fireflies(opt);

        // Update best solution
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
        }

        // Optional: Log progress
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }
}
