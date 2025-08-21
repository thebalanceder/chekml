#include "POA.h"
#include <string.h>  // For memcpy

// Initialize Population
static inline __attribute__((always_inline)) void initialize_population_poa(Optimizer *restrict opt) {
    // ✅ Initialize population randomly within bounds
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    Solution *restrict population = opt->population;
    const double *restrict bounds = opt->bounds;

    for (int i = 0; i < pop_size; i++) {
        double *pos = population[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] = rand_double_poa(bounds[2 * j], bounds[2 * j + 1]);
        }
        population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void POA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *restrict opt = (Optimizer *)opt_void;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    Solution *restrict population = opt->population;
    Solution *restrict best_solution = &opt->best_solution;
    const double *restrict bounds = opt->bounds;

    // ✅ Initialize population
    initialize_population_poa(opt);

    // ✅ Pre-allocate direction array
    double direction[dim];

    // ✅ Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness and update best solution
        for (int i = 0; i < pop_size; i++) {
            double *pos = population[i].position;
            double f = objective_function(pos);  // Use provided objective function
            population[i].fitness = f;

            if (f < best_solution->fitness) {
                best_solution->fitness = f;
                memcpy(best_solution->position, pos, dim * sizeof(double));
            }
        }

        // Update positions (scalar code for debugging)
        for (int i = 0; i < pop_size; i++) {
            double *pos = population[i].position;
            double norm = 0.0;

            // Compute direction and norm
            for (int j = 0; j < dim; j++) {
                direction[j] = best_solution->position[j] - pos[j];
                norm += direction[j] * direction[j];
            }
            norm = sqrt(norm);

            // Update position
            if (norm > 0.0) {
                double inv_norm = STEP_SIZE / norm;
                for (int j = 0; j < dim; j++) {
                    pos[j] += direction[j] * inv_norm;
                    // Inline bounds checking
                    if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
                    else if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
                }
            }
        }

        // Log iteration progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, best_solution->fitness);
    }
}
