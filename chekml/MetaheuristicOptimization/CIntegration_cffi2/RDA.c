/* RDA.c - Optimized Implementation file for Red Deer Algorithm (RDA) Optimization */
#include "RDA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator

// Main Optimization Function with Performance Optimizations
void RDA_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double step_size_2 = 2.0 * STEP_SIZE;  // Precompute for exploration phase
    double *restrict best_pos = opt->best_solution.position;
    double *restrict best_fitness = &opt->best_solution.fitness;
    double *restrict bounds = opt->bounds;

    // Seed the random number generator (do this once outside the loop)
    srand((unsigned int)time(NULL));

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Process each individual in the population
        for (int i = 0; i < pop_size; i++) {
            double *restrict pos = opt->population[i].position;
            double r = rand() * INV_RAND_MAX;

            // Update position (inlined exploration/exploitation logic)
            if (r < P_EXPLORATION) {
                // Exploration: Move randomly
                for (int j = 0; j < dim; j++) {
                    pos[j] += STEP_SIZE * (rand() * INV_RAND_MAX * step_size_2 - 1.0);
                    // Enforce bounds in-place
                    pos[j] = (pos[j] < bounds[2 * j]) ? bounds[2 * j] : 
                             (pos[j] > bounds[2 * j + 1]) ? bounds[2 * j + 1] : pos[j];
                }
            } else {
                // Exploitation: Move towards the best solution
                for (int j = 0; j < dim; j++) {
                    pos[j] += STEP_SIZE * (best_pos[j] - pos[j]);
                    // Enforce bounds in-place
                    pos[j] = (pos[j] < bounds[2 * j]) ? bounds[2 * j] : 
                             (pos[j] > bounds[2 * j + 1]) ? bounds[2 * j + 1] : pos[j];
                }
            }

            // Evaluate fitness immediately after updating position
            double new_fitness = objective_function(pos);
            opt->population[i].fitness = new_fitness;

            // Update best solution if the new fitness is better
            if (new_fitness < *best_fitness) {
                *best_fitness = new_fitness;
                for (int j = 0; j < dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }

        // Log progress (keep this lightweight)
        printf("Iteration %d: Best Value = %f\n", iter + 1, *best_fitness);
    }
}
