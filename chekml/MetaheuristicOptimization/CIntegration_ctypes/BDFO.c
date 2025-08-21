#include "BDFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and RAND_MAX
#include <time.h>    // For time() to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Bottom Grubbing Phase (Mimics dolphin foraging behavior)
void bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *)) {
    int half_dim = (int)(opt->dim * SEGMENT_FACTOR);
    double *perturbed1 = (double *)malloc(opt->dim * sizeof(double));
    double *perturbed2 = (double *)malloc(opt->dim * sizeof(double));
    double *perturbation = (double *)malloc(opt->dim * sizeof(double));

    if (!perturbed1 || !perturbed2 || !perturbation) {
        fprintf(stderr, "Memory allocation failed in bottom_grubbing_phase\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        // Copy current solution
        double current_fitness = objective_function(opt->population[i].position);

        // Generate perturbation
        for (int j = 0; j < opt->dim; j++) {
            perturbation[j] = rand_double(-BDFO_PERTURBATION_SCALE, BDFO_PERTURBATION_SCALE);
            perturbed1[j] = opt->population[i].position[j];
            perturbed2[j] = opt->population[i].position[j];
        }

        // Perturb first segment
        for (int j = 0; j < half_dim; j++) {
            perturbed1[j] += perturbation[j];
        }
        double fitness1 = objective_function(perturbed1);

        // Perturb second segment
        for (int j = half_dim; j < opt->dim; j++) {
            perturbed2[j] += perturbation[j];
        }
        double fitness2 = objective_function(perturbed2);

        // Adjust solution
        if (opt->best_solution.fitness != INFINITY) {
            for (int j = 0; j < half_dim; j++) {
                if (fitness1 < current_fitness) {
                    opt->population[i].position[j] += ADJUSTMENT_RATE * 
                        (opt->best_solution.position[j] - opt->population[i].position[j]);
                }
            }
            for (int j = half_dim; j < opt->dim; j++) {
                if (fitness2 < current_fitness) {
                    opt->population[i].position[j] += ADJUSTMENT_RATE * 
                        (opt->best_solution.position[j] - opt->population[i].position[j]);
                }
            }
        } else {
            // Blend with a random solution
            int random_idx = (int)rand_double(0, opt->population_size);
            for (int j = 0; j < half_dim; j++) {
                opt->population[i].position[j] += ADJUSTMENT_RATE * 
                    (opt->population[random_idx].position[j] - opt->population[i].position[j]);
            }
            for (int j = half_dim; j < opt->dim; j++) {
                opt->population[i].position[j] += ADJUSTMENT_RATE * 
                    (opt->population[random_idx].position[j] - opt->population[i].position[j]);
            }
        }
    }

    free(perturbed1);
    free(perturbed2);
    free(perturbation);
    enforce_bound_constraints(opt);
}

// Exploration Phase (Random perturbations for diversity)
void bdfo_exploration_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) < EXPLORATION_PROB) {
            for (int j = 0; j < opt->dim; j++) {
                double perturbation = EXPLORATION_FACTOR * rand_double(-1.0, 1.0) * 
                                     (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
                opt->population[i].position[j] += perturbation;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Elimination Phase (Replace worst solutions)
void elimination_phase_bdfo(Optimizer *opt) {
    int worst_count = (int)(ELIMINATION_RATIO * opt->population_size);
    
    for (int i = 0; i < worst_count; i++) {
        int worst_idx = opt->population_size - i - 1;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[worst_idx].position[j] = opt->bounds[2 * j] + 
                (rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
        }
        opt->population[worst_idx].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update fitness for all solutions
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Execute algorithm phases
        bottom_grubbing_phase(opt, objective_function);
        bdfo_exploration_phase(opt);
        elimination_phase_bdfo(opt);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
