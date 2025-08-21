#include "BDFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Bottom Grubbing Phase (Optimized)
void bdfo_bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int half_dim = (int)(opt->dim * BDFO_SEGMENT_FACTOR);
    int dim = opt->dim;
    int pop_size = opt->population_size;

    // Allocate temporary arrays once
    double *perturbed1 = (double *)malloc(dim * sizeof(double));
    double *perturbed2 = (double *)malloc(dim * sizeof(double));
    double *perturbation = (double *)malloc(dim * sizeof(double));
    if (!perturbed1 || !perturbed2 || !perturbation) {
        fprintf(stderr, "Memory allocation failed in bdfo_bottom_grubbing_phase\n");
        free(perturbed1); free(perturbed2); free(perturbation);
        return;
    }

    // Cache best solution fitness
    double best_fitness = opt->best_solution.fitness;
    double *best_pos = opt->best_solution.position;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double current_fitness = objective_function(pos);

        // Generate perturbations
        for (int j = 0; j < dim; j++) {
            perturbation[j] = bdfo_rand_double(-BDFO_PERTURBATION_SCALE, BDFO_PERTURBATION_SCALE);
            perturbed1[j] = pos[j];
            perturbed2[j] = pos[j];
        }

        // Perturb first segment
        for (int j = 0; j < half_dim; j++) {
            perturbed1[j] += perturbation[j];
        }
        double fitness1 = objective_function(perturbed1);

        // Perturb second segment
        for (int j = half_dim; j < dim; j++) {
            perturbed2[j] += perturbation[j];
        }
        double fitness2 = objective_function(perturbed2);

        // Adjust solution based on fitness improvement
        if (best_fitness != INFINITY) {
            for (int j = 0; j < dim; j++) {
                double adjustment = BDFO_ADJUSTMENT_RATE * (best_pos[j] - pos[j]);
                if (j < half_dim && fitness1 < current_fitness) {
                    pos[j] += adjustment;
                } else if (j >= half_dim && fitness2 < current_fitness) {
                    pos[j] += adjustment;
                }
            }
        } else {
            int random_idx = (int)bdfo_rand_double(0, pop_size);
            double *rand_pos = opt->population[random_idx].position;
            for (int j = 0; j < dim; j++) {
                pos[j] += BDFO_ADJUSTMENT_RATE * (rand_pos[j] - pos[j]);
            }
        }
    }

    free(perturbed1);
    free(perturbed2);
    free(perturbation);
    enforce_bound_constraints(opt);
}

// Exploration Phase (Optimized)
void bdfo_exploration_phase(Optimizer *opt) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;

    for (int i = 0; i < pop_size; i++) {
        if (bdfo_rand_double(0.0, 1.0) < BDFO_EXPLORATION_PROB) {
            double *pos = opt->population[i].position;
            for (int j = 0; j < dim; j++) {
                double range = bounds[2 * j + 1] - bounds[2 * j];
                pos[j] += BDFO_EXPLORATION_FACTOR * bdfo_rand_double(-1.0, 1.0) * range;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Elimination Phase (Optimized)
void bdfo_elimination_phase(Optimizer *opt) {
    int worst_count = (int)(BDFO_ELIMINATION_RATIO * opt->population_size);
    int dim = opt->dim;
    double *bounds = opt->bounds;

    // Assume population is sorted or worst solutions are at the end
    for (int i = 0; i < worst_count; i++) {
        int idx = opt->population_size - i - 1;
        double *pos = opt->population[idx].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = lower + bdfo_rand_double(0.0, 1.0) * (upper - lower);
        }
        opt->population[idx].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    double *best_pos = opt->best_solution.position;
    double *fitness_values = (double *)malloc(pop_size * sizeof(double));
    if (!fitness_values) {
        fprintf(stderr, "Memory allocation failed in BDFO_optimize\n");
        return;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness and update best solution
        double best_fitness = opt->best_solution.fitness;
        int best_idx = -1;
        for (int i = 0; i < pop_size; i++) {
            double *pos = opt->population[i].position;
            fitness_values[i] = objective_function(pos);
            if (fitness_values[i] < best_fitness) {
                best_fitness = fitness_values[i];
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            opt->best_solution.fitness = best_fitness;
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = opt->population[best_idx].position[j];
            }
        }

        // Execute algorithm phases
        bdfo_bottom_grubbing_phase(opt, objective_function);
        bdfo_exploration_phase(opt);
        bdfo_elimination_phase(opt);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(fitness_values);
}
