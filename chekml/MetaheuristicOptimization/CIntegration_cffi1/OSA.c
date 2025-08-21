#include "OSA.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>

// Fast initialization of population within bounds
static inline void initialize_population(Optimizer *restrict opt, unsigned int *seed) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *restrict bounds = opt->bounds;

    for (int i = 0; i < pop_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * osa_fast_rand(0.0, 1.0, seed);
        }
        opt->population[i].fitness = INFINITY;
    }
}

// Exploration Phase (Random Movement)
void osa_exploration_phase(Optimizer *restrict opt, int index, double *restrict bounds, unsigned int *seed) {
    const int dim = opt->dim;
    double *restrict pos = opt->population[index].position;

    for (int j = 0; j < dim; j++) {
        double random_move = OSA_STEP_SIZE * (2.0 * osa_fast_rand(0.0, 1.0, seed) - 1.0);
        double new_pos = pos[j] + random_move;
        double lb = bounds[2 * j];
        double ub = bounds[2 * j + 1];
        pos[j] = fmax(lb, fmin(ub, new_pos));
    }
}

// Exploitation Phase (Move towards best solution)
void osa_exploitation_phase(Optimizer *restrict opt, int index, const double *restrict best_pos, double *restrict bounds, unsigned int *seed) {
    const int dim = opt->dim;
    double *restrict pos = opt->population[index].position;

    for (int j = 0; j < dim; j++) {
        double direction = best_pos[j] - pos[j];
        double new_pos = pos[j] + OSA_STEP_SIZE * direction;
        double lb = bounds[2 * j];
        double ub = bounds[2 * j + 1];
        pos[j] = fmax(lb, fmin(ub, new_pos));
    }
}

// Main Optimization Function
void OSA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    unsigned int seed = (unsigned int)time(NULL); // Simple seed initialization
    initialize_population(opt, &seed);

    const int pop_size = opt->population_size;
    const int max_iter = opt->max_iter;
    const int dim = opt->dim;
    double *restrict bounds = opt->bounds;
    double *restrict best_pos = opt->best_solution.position;
    double best_fitness = opt->best_solution.fitness;

    // Precompute exploration probability threshold
    const double explore_threshold = OSA_P_EXPLORE;

    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness and update best solution
        for (int i = 0; i < pop_size; i++) {
            double *restrict pos = opt->population[i].position;
            double new_fitness = objective_function(pos);
            opt->population[i].fitness = new_fitness;

            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                memcpy(best_pos, pos, dim * sizeof(double));
            }
        }
        opt->best_solution.fitness = best_fitness;

        // Update positions
        for (int i = 0; i < pop_size; i++) {
            if (osa_fast_rand(0.0, 1.0, &seed) < explore_threshold) {
                osa_exploration_phase(opt, i, bounds, &seed);
            } else {
                osa_exploitation_phase(opt, i, best_pos, bounds, &seed);
            }
        }

        // Log iteration progress (optional, can be disabled for max speed)
        printf("Iteration %d: Best Value = %f\n", iter + 1, best_fitness);
    }
}
