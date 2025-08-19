/* BSO.c - Optimized Implementation file for Buffalo Swarm Optimization */
#include "BSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand()
#include <time.h>    // For time() to seed random generator

// Initialize Population
void initialize_population_bso(Optimizer *opt) {
    // Cache bounds for faster access
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = rand_double_bso(lower, upper);
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness
    }
    // Defer bounds enforcement to first objective function evaluation
}

// Local Search Phase with Integrated Bounds Enforcement
void local_search(Optimizer *opt) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            // Apply random perturbation (approximating normal distribution)
            double perturbation = PERTURBATION_SCALE_BSO * (rand_double_bso(-1.0, 1.0));
            pos[j] += perturbation;
            // Inline bounds clipping
            pos[j] = CLIP(pos[j], bounds[2 * j], bounds[2 * j + 1]);
        }
    }
    // No need for separate enforce_bound_constraints call
}

// Main Optimization Function
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator (should be done once in main program)
    // srand(time(NULL)); // Uncomment if needed

    // Initialize population
    initialize_population_bso(opt);

    // Cache frequently used variables
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_pos = opt->best_solution.position;
    double *best_fitness = &opt->best_solution.fitness;

    // Set initial best solution
    *best_fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double fitness = objective_function(pos);
        opt->population[i].fitness = fitness;
        if (fitness < *best_fitness) {
            *best_fitness = fitness;
            for (int j = 0; j < dim; j++) {
                best_pos[j] = pos[j];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Perform local search with integrated bounds enforcement
        local_search(opt);

        // Update fitness and best solution
        for (int i = 0; i < pop_size; i++) {
            double *pos = opt->population[i].position;
            double fitness = objective_function(pos);
            opt->population[i].fitness = fitness;
            if (fitness < *best_fitness) {
                *best_fitness = fitness;
                for (int j = 0; j < dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }
    }
}
