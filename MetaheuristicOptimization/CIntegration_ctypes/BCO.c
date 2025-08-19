/* BCO.c - Implementation file for Bacterial Colony Optimization */
#include "BCO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a Gaussian random number (simplified Box-Muller)
double rand_gaussian() {
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Compute adaptive chemotaxis step size
double compute_chemotaxis_step(int iteration, int max_iter) {
    return CHEMOTAXIS_STEP_MIN + (CHEMOTAXIS_STEP_MAX - CHEMOTAXIS_STEP_MIN) * 
           ((double)(max_iter - iteration) / max_iter);
}

// Chemotaxis and Communication Phase
void chemotaxis_and_communication(Optimizer *opt, int iteration) {
    double chemotaxis_step = compute_chemotaxis_step(iteration, opt->max_iter);
    double *new_position = (double *)malloc(opt->dim * sizeof(double));
    double *turbulent = (double *)malloc(opt->dim * sizeof(double));
    
    for (int i = 0; i < opt->population_size; i++) {
        double r = rand_double(0.0, 1.0);
        if (r < 0.5) {
            // Tumbling (exploration with random direction)
            for (int j = 0; j < opt->dim; j++) {
                turbulent[j] = rand_gaussian();
                new_position[j] = opt->population[i].position[j] + chemotaxis_step * (
                    0.5 * (opt->best_solution.position[j] - opt->population[i].position[j]) +
                    turbulent[j]
                );
            }
        } else {
            // Swimming (exploitation towards best solution)
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = opt->population[i].position[j] + chemotaxis_step * (
                    0.5 * (opt->best_solution.position[j] - opt->population[i].position[j])
                );
            }
        }

        // Communication
        if (rand_double(0.0, 1.0) < COMMUNICATION_PROB) {
            int neighbor_idx;
            if (rand_double(0.0, 1.0) < 0.5) {
                // Dynamic neighbor oriented
                neighbor_idx = (i + (rand() % 2 == 0 ? -1 : 1)) % opt->population_size;
                if (neighbor_idx < 0) neighbor_idx += opt->population_size;
            } else {
                // Random oriented
                neighbor_idx = rand() % opt->population_size;
            }

            double neighbor_fitness = opt->population[neighbor_idx].fitness;
            if (neighbor_fitness < opt->population[i].fitness) {
                memcpy(new_position, opt->population[neighbor_idx].position, opt->dim * sizeof(double));
            } else if (opt->population[i].fitness > opt->best_solution.fitness) {
                memcpy(new_position, opt->best_solution.position, opt->dim * sizeof(double));
            }
        }

        // Update position
        memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
    }

    free(new_position);
    free(turbulent);
    enforce_bound_constraints(opt);
}

// Elimination and Reproduction Phase
void elimination_and_reproduction(Optimizer *opt) {
    // Compute energy levels based on fitness
    double *energy_levels = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        energy_levels[i] = 1.0 / (1.0 + opt->population[i].fitness);
    }

    // Sort indices by fitness
    int *sorted_indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        sorted_indices[i] = i;
    }
    // Simple bubble sort for simplicity (could be optimized with qsort)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[sorted_indices[j]].fitness > opt->population[sorted_indices[j + 1]].fitness) {
                int temp = sorted_indices[j];
                sorted_indices[j] = sorted_indices[j + 1];
                sorted_indices[j + 1] = temp;
            }
        }
    }

    // Elimination
    int num_eliminate = (int)(ELIMINATION_RATIO_BCO * opt->population_size);
    for (int i = 0; i < num_eliminate; i++) {
        int idx = sorted_indices[opt->population_size - 1 - i];
        if (energy_levels[idx] < REPRODUCTION_THRESHOLD) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[idx].position[j] = opt->bounds[2 * j] + 
                                                  (rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
            }
            opt->population[idx].fitness = INFINITY;
        }
    }

    // Reproduction
    int num_reproduce = num_eliminate / 2;
    for (int i = 0; i < num_reproduce; i++) {
        int idx = sorted_indices[i];
        if (energy_levels[idx] >= REPRODUCTION_THRESHOLD) {
            memcpy(opt->population[sorted_indices[opt->population_size - 1 - i]].position,
                   opt->population[idx].position, opt->dim * sizeof(double));
        }
    }

    free(energy_levels);
    free(sorted_indices);
    enforce_bound_constraints(opt);
}

// Migration Phase
void migration_phase(Optimizer *opt) {
    double *energy_levels = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        energy_levels[i] = 1.0 / (1.0 + opt->population[i].fitness);
    }

    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) < MIGRATION_PROBABILITY) {
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                norm += pow(opt->population[i].position[j] - opt->best_solution.position[j], 2);
            }
            norm = sqrt(norm);
            if (energy_levels[i] < REPRODUCTION_THRESHOLD || norm < 1e-3) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = opt->bounds[2 * j] + 
                                                    (rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
                }
                opt->population[i].fitness = INFINITY;
            }
        }
    }

    free(energy_levels);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BCO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize best solution
    for (int i = 0; i < opt->dim; i++) {
        opt->best_solution.position[i] = opt->population[0].position[i];
    }
    opt->best_solution.fitness = objective_function(opt->best_solution.position);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update fitness for all bacteria
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        chemotaxis_and_communication(opt, iter);
        elimination_and_reproduction(opt);
        migration_phase(opt);

        // Debugging output
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
