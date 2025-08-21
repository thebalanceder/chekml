#include "OSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Initialize population randomly within bounds
static void initialize_population(Optimizer *restrict opt) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *bounds = opt->bounds;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * rand_double_osa(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
    // Bounds enforced during initialization
}

// Exploration Phase (Random Movement)
void osa_exploration_phase(Optimizer *restrict opt, int index, double *restrict bounds) {
    const int dim = opt->dim;
    double *pos = opt->population[index].position;

    for (int j = 0; j < dim; j++) {
        double random_move = OSA_STEP_SIZE * (2.0 * rand_double_osa(0.0, 1.0) - 1.0);
        double new_pos = pos[j] + random_move;
        double lb = bounds[2 * j];
        double ub = bounds[2 * j + 1];
        pos[j] = fmax(lb, fmin(ub, new_pos)); // Inline bounds checking
    }
}

// Exploitation Phase (Move towards best solution)
void osa_exploitation_phase(Optimizer *restrict opt, int index, const double *restrict best_pos, double *restrict bounds) {
    const int dim = opt->dim;
    double *pos = opt->population[index].position;

    for (int j = 0; j < dim; j++) {
        double direction = best_pos[j] - pos[j];
        double new_pos = pos[j] + OSA_STEP_SIZE * direction;
        double lb = bounds[2 * j];
        double ub = bounds[2 * j + 1];
        pos[j] = fmax(lb, fmin(ub, new_pos)); // Inline bounds checking
    }
}

// Main Optimization Function
void OSA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    // Initialize population
    initialize_population(opt);

    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    const int dim = opt->dim;
    double *bounds = opt->bounds;
    double *best_pos = opt->best_solution.position;
    double best_fitness = opt->best_solution.fitness;

    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness and update best solution
        for (int i = 0; i < pop_size; i++) {
            double *pos = opt->population[i].position;
            double new_fitness = objective_function(pos);
            opt->population[i].fitness = new_fitness;

            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                for (int j = 0; j < dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }
        opt->best_solution.fitness = best_fitness;

        // Update each individual's position
        for (int i = 0; i < pop_size; i++) {
            if (rand_double_osa(0.0, 1.0) < OSA_P_EXPLORE) {
                osa_exploration_phase(opt, i, bounds);
            } else {
                osa_exploitation_phase(opt, i, best_pos, bounds);
            }
        }

        // Log iteration progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, best_fitness);
    }
}
