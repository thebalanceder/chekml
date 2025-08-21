/* BSO.c - Implementation file for Buffalo Swarm Optimization */
#include "BSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void initialize_population_bso(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Assume bounds are stored as [lower0, upper0, lower1, upper1, ...]
            double lower = opt->bounds[2 * j];
            double upper = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = rand_double(lower, upper);
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness
    }
    enforce_bound_constraints(opt);
}

// Local Search Phase
void local_search(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Apply random perturbation
            double perturbation = PERTURBATION_SCALE_BSO * ((double)rand() / RAND_MAX - 0.5) * 2.0;  // Normal-like distribution
            opt->population[i].position[j] += perturbation;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize population
    initialize_population_bso(opt);

    // Set initial best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Perform local search
        local_search(opt);

        // Update fitness and best solution
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }
}
