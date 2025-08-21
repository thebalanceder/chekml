#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

// Random double in range [min, max]
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static Solution nearest_neighbor(Solution *population, int pop_size, double *target_pos, int dim, int skip_idx) {
    Solution nn;
    double min_dist = DBL_MAX;

    for (int i = 0; i < pop_size; i++) {
        if (i == skip_idx) continue;
        double dist = 0.0;
        for (int d = 0; d < dim; d++) {
            dist += (population[i].position[d] - target_pos[d]) * (population[i].position[d] - target_pos[d]);
        }
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            nn = population[i];
        }
    }

    return nn;
}

static void swirl(double *candidate, double *target_pos, double *nn_pos, int dim, int S, int s_max, double *lb, double *ub) {
    double swirl_strength = (s_max - S + 1) / (double)s_max;
    for (int i = 0; i < dim; i++) {
        candidate[i] = target_pos[i] + swirl_strength * (nn_pos[i] - target_pos[i]) * rand_double(-1, 1);

        // Boundary enforcement
        if (candidate[i] < lb[i]) candidate[i] = lb[i];
        if (candidate[i] > ub[i]) candidate[i] = ub[i];
    }
}

void KA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    double *lb = opt->bounds;  // Lower bounds
    double *ub = &opt->bounds[dim]; // Upper bounds

    // Initial population fitness evaluation
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            double r = rand_double(0.0, 1.0);
            int partner_idx = rand() % pop_size;

            double trial[dim];
            for (int j = 0; j < dim; j++) {
                trial[j] = opt->population[i].position[j] +
                          r * (opt->population[partner_idx].position[j] - opt->population[i].position[j]);

                // Boundary enforcement
                if (trial[j] < lb[j]) trial[j] = lb[j];
                if (trial[j] > ub[j]) trial[j] = ub[j];
            }

            double trial_fitness = objective_function(trial);
            if (trial_fitness < opt->population[i].fitness) {
                memcpy(opt->population[i].position, trial, sizeof(double) * dim);
                opt->population[i].fitness = trial_fitness;

                if (trial_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = trial_fitness;
                    memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                }
            }
        }
        
        enforce_bound_constraints(opt);  // Use your existing function for boundary constraints
    }
}
