#include "SPO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Define constants specific to SPO
#define MAX_ITER 100
#define PAINT_FACTOR 0.1
#define INF 1e30  // Infinity value for uninitialized fitness

// Random number generator between min and max
double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Update the population using SPO's unique movement mechanism
void update_population_spo(Optimizer* opt) {
    int best_index = 0;
    
    // Find best solution
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_index].fitness) {
            best_index = i;
        }
    }

    // Copy the best solution to the best_solution of optimizer
    memcpy(opt->best_solution.position, opt->population[best_index].position, opt->dim * sizeof(double));
    opt->best_solution.fitness = opt->population[best_index].fitness;

    // Update other solutions (movement influenced by the best solution)
    for (int i = 0; i < opt->population_size; i++) {
        if (i != best_index) {
            // SPO movement mechanism
            double rand_factor = rand_uniform(0, 1);  // A factor for stochastic movement
            for (int d = 0; d < opt->dim; d++) {
                double movement = rand_factor * (opt->best_solution.position[d] - opt->population[i].position[d]);
                opt->population[i].position[d] += PAINT_FACTOR * movement;  // Stochastic 'paint' influence
            }
        }
    }
}

// Evaluate the fitness of each individual in the population
void evaluate_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Main SPO optimization loop
void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = rand_uniform(min_bound, max_bound);
        }
        opt->population[i].fitness = INF;  // Start with high fitness
    }

    // Optimization loop
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Evaluate the population
        evaluate_population_spo(opt, objective_function);

        // Update the population using SPO movement
        update_population_spo(opt);

        // Enforce boundaries on the population (if any)
        enforce_bound_constraints(opt);

        // Optionally print progress
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }
}

