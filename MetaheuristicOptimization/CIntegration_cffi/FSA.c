#include "FSA.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy()
#include <math.h>    // For INFINITY
#if FSA_DEBUG
#include <stdio.h>   // For logging
#endif

// Additional data for FSA
typedef struct {
    double **local_best_positions;  // Local best positions for each solution
    double *local_best_values;      // Local best fitness values
} FSAData;

// Inline random number generator
#define RAND_DOUBLE(min, max) ((min) + ((max) - (min)) * ((double)rand() / RAND_MAX))

// Internal helper functions
static void fsa_initialize_population(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function, double *temp_solution) {
    int i, j;
    double fitness;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Allocate memory for FSAData
    data->local_best_positions = (double **)malloc(opt->population_size * sizeof(double *));
    data->local_best_values = (double *)malloc(opt->population_size * sizeof(double));
    for (i = 0; i < opt->population_size; i++) {
        data->local_best_positions[i] = (double *)malloc(opt->dim * sizeof(double));
    }

    // Initialize population randomly within bounds
    for (i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        double *local_best = data->local_best_positions[i];
        for (j = 0; j < opt->dim; j++) {
            position[j] = lower_bounds[2 * j] + RAND_DOUBLE(0.0, 1.0) * (upper_bounds[2 * j] - lower_bounds[2 * j]);
            local_best[j] = position[j];
        }
        fitness = objective_function(position);
        opt->population[i].fitness = fitness;
        data->local_best_values[i] = fitness;
    }

    // Find initial global best
    int best_idx = 0;
    for (i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));

    enforce_bound_constraints(opt);
}

// Internal helper function for population update
static void fsa_update_population(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function, double *temp_solution) {
    int i, j;
    double new_fitness;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;
    double *global_best = opt->best_solution.position;

    for (i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        double *local_best = data->local_best_positions[i];

        // Update rule: Move towards global and local bests
        for (j = 0; j < opt->dim; j++) {
            double delta_global = (global_best[j] - position[j]) * RAND_DOUBLE(0.0, 1.0);
            double delta_local = (local_best[j] - position[j]) * RAND_DOUBLE(0.0, 1.0);
            temp_solution[j] = position[j] + delta_global + delta_local;

            // Enforce bounds
            if (temp_solution[j] < lower_bounds[2 * j]) {
                temp_solution[j] = lower_bounds[2 * j];
            } else if (temp_solution[j] > upper_bounds[2 * j]) {
                temp_solution[j] = upper_bounds[2 * j];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(temp_solution);

        // Update local best
        if (new_fitness <= data->local_best_values[i]) {
            memcpy(data->local_best_positions[i], temp_solution, opt->dim * sizeof(double));
            data->local_best_values[i] = new_fitness;
        }

        // Update global best
        if (new_fitness <= opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(opt->best_solution.position, temp_solution, opt->dim * sizeof(double));
        }

        // Update population
        memcpy(position, temp_solution, opt->dim * sizeof(double));
        opt->population[i].fitness = new_fitness;
    }

    enforce_bound_constraints(opt);
}

// Public interface for population update
void update_population_fsa(Optimizer *opt) {
#if FSA_DEBUG
    fprintf(stderr, "Error: update_population_fsa should be called via FSA_optimize\n");
#endif
    exit(1);
}

// Internal helper function for initial strategy update
static void fsa_update_with_initial_strategy(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function, double *temp_solution, double *temp_fitness) {
    int i, j;
    double new_fitness;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;
    double *global_best = opt->best_solution.position;

    // Create temporary population
    for (i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        double *local_best = data->local_best_positions[i];

        for (j = 0; j < opt->dim; j++) {
            temp_solution[j] = global_best[j] + (global_best[j] - local_best[j]) * RAND_DOUBLE(0.0, 1.0);

            // Enforce bounds
            if (temp_solution[j] < lower_bounds[2 * j]) {
                temp_solution[j] = lower_bounds[2 * j];
            } else if (temp_solution[j] > upper_bounds[2 * j]) {
                temp_solution[j] = upper_bounds[2 * j];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(temp_solution);
        temp_fitness[i] = new_fitness;

        // Update if better than previous fitness
        if (new_fitness <= data->local_best_values[i]) {
            memcpy(position, temp_solution, opt->dim * sizeof(double));
            memcpy(local_best, temp_solution, opt->dim * sizeof(double));
            data->local_best_values[i] = new_fitness;
            opt->population[i].fitness = new_fitness;
        }
    }

    // Update global best based on temporary population
    int best_idx = 0;
    for (i = 1; i < opt->population_size; i++) {
        if (temp_fitness[i] < temp_fitness[best_idx]) {
            best_idx = i;
        }
    }
    if (temp_fitness[best_idx] <= opt->best_solution.fitness) {
        opt->best_solution.fitness = temp_fitness[best_idx];
        memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
    }

    enforce_bound_constraints(opt);
}

// Public interface for initial strategy update
void update_with_initial_strategy(Optimizer *opt) {
#if FSA_DEBUG
    fprintf(stderr, "Error: update_with_initial_strategy should be called via FSA_optimize\n");
#endif
    exit(1);
}

// Main Optimization Function
void FSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int run, iter;
    double *best_scores = (double *)malloc(FSA_NUM_RUNS * sizeof(double));
    double **best_positions = (double **)malloc(FSA_NUM_RUNS * sizeof(double *));
    double *temp_solution = (double *)malloc(opt->dim * sizeof(double));
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));

    for (run = 0; run < FSA_NUM_RUNS; run++) {
        best_positions[run] = (double *)malloc(opt->dim * sizeof(double));
    }

    for (run = 0; run < FSA_NUM_RUNS; run++) {
        // Allocate FSAData for this run
        FSAData data;
        fsa_initialize_population(opt, &data, objective_function, temp_solution);

        for (iter = 0; iter < opt->max_iter; iter++) {
            fsa_update_population(opt, &data, objective_function, temp_solution);
            fsa_update_with_initial_strategy(opt, &data, objective_function, temp_solution, temp_fitness);
#if FSA_DEBUG
            printf("Run %d, Iteration %d: Best Value = %f\n", run + 1, iter + 1, opt->best_solution.fitness);
#endif
        }

        best_scores[run] = opt->best_solution.fitness;
        memcpy(best_positions[run], opt->best_solution.position, opt->dim * sizeof(double));

        // Clean up FSAData for this run
        for (int i = 0; i < opt->population_size; i++) {
            free(data.local_best_positions[i]);
        }
        free(data.local_best_positions);
        free(data.local_best_values);
    }

    // Find the best result across all runs
    int best_run = 0;
    for (run = 1; run < FSA_NUM_RUNS; run++) {
        if (best_scores[run] < best_scores[best_run]) {
            best_run = run;
        }
    }
    opt->best_solution.fitness = best_scores[best_run];
    memcpy(opt->best_solution.position, best_positions[best_run], opt->dim * sizeof(double));

#if FSA_DEBUG
    printf("Best Score across all runs: %f\n", opt->best_solution.fitness);
    printf("Best Position: ");
    for (int j = 0; j < opt->dim; j++) {
        printf("%f ", opt->best_solution.position[j]);
    }
    printf("\n");
#endif

    // Clean up
    for (run = 0; run < FSA_NUM_RUNS; run++) {
        free(best_positions[run]);
    }
    free(best_positions);
    free(best_scores);
    free(temp_solution);
    free(temp_fitness);
}
