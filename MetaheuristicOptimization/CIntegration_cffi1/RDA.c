/* RDA.c - Extreme Speed Implementation for Red Deer Algorithm (RDA) Optimization */
#include "RDA.h"
#include <stdlib.h>

// Main Optimization Function
void RDA_optimize(void* optimizer, ObjectiveFunction objective_function) {
    // Cast the void* to Optimizer*
    Optimizer* opt = (Optimizer*)optimizer;
    
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    double* bounds = opt->bounds;
    Solution* population = opt->population;
    double* best_pos = opt->best_solution.position;
    double* best_fitness = &opt->best_solution.fitness;

    // Allocate RDAData for additional data
    RDAData data;
    data.fitness = (double*)malloc(pop_size * sizeof(double));
    if (!data.fitness) {
        printf("Memory allocation failed for fitness array\n");
        return;
    }

    // Initialize Xorshift RNG (seed with a non-zero value)
    data.rng_state.a = 123456789 ^ (uint32_t)(*best_fitness);

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Process each individual in the population
        for (int i = 0; i < pop_size; i++) {
            double* pos = population[i].position;
            double r = RAND_DOUBLE(&data.rng_state);

            // Exploration or Exploitation (branchless)
            const double is_exploration = (r < P_EXPLORATION) ? 1.0 : 0.0;
            const double is_exploitation = 1.0 - is_exploration;

            // Update position
            for (int j = 0; j < dim; j++) {
                double exploration_step = STEP_SIZE * (RAND_DOUBLE(&data.rng_state) * STEP_SIZE_2 - 1.0);
                double exploitation_step = STEP_SIZE * (best_pos[j] - pos[j]);
                pos[j] += is_exploration * exploration_step + is_exploitation * exploitation_step;
                pos[j] = CLAMP(pos[j], bounds[2 * j], bounds[2 * j + 1]);
            }

            // Evaluate fitness
            double new_fitness = objective_function(pos);
            data.fitness[i] = new_fitness;

            // Update best solution (branchless)
            const double better = (new_fitness < *best_fitness) ? 1.0 : 0.0;
            *best_fitness = better * new_fitness + (1.0 - better) * (*best_fitness);
            for (int j = 0; j < dim; j++) {
                best_pos[j] = better * pos[j] + (1.0 - better) * best_pos[j];
            }
        }

        // Log progress (minimal)
        if (iter % 10 == 0) {
            printf("Iter %d: Best = %.6f\n", iter + 1, *best_fitness);
        }
    }

    // Clean up
    free(data.fitness);
}
