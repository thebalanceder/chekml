#include "FSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy()

// Additional data for FSA
typedef struct {
    double **local_best_positions;  // Local best positions for each solution
    double *local_best_values;      // Local best fitness values
} FSAData;

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Internal helper functions
static void fsa_initialize_population(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function) {
    int i, j;
    double fitness;

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
        for (j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                             rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            data->local_best_positions[i][j] = opt->population[i].position[j];
        }
        fitness = objective_function(opt->population[i].position);
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
    for (j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    enforce_bound_constraints(opt);
}

// Internal helper function for population update
static void fsa_update_population(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function) {
    int i, j;
    double new_fitness;
    double *new_solution = (double *)malloc(opt->dim * sizeof(double));

    for (i = 0; i < opt->population_size; i++) {
        // Update rule: Move towards global and local bests
        for (j = 0; j < opt->dim; j++) {
            new_solution[j] = opt->population[i].position[j] +
                              (-opt->population[i].position[j] + opt->best_solution.position[j]) * rand_double(0.0, 1.0) +
                              (-opt->population[i].position[j] + data->local_best_positions[i][j]) * rand_double(0.0, 1.0);
        }

        // Enforce bounds
        for (j = 0; j < opt->dim; j++) {
            if (new_solution[j] < opt->bounds[2 * j]) {
                new_solution[j] = opt->bounds[2 * j];
            } else if (new_solution[j] > opt->bounds[2 * j + 1]) {
                new_solution[j] = opt->bounds[2 * j + 1];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(new_solution);

        // Update local best
        if (new_fitness <= data->local_best_values[i]) {
            for (j = 0; j < opt->dim; j++) {
                data->local_best_positions[i][j] = new_solution[j];
            }
            data->local_best_values[i] = new_fitness;
        }

        // Update global best
        if (new_fitness <= opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = new_solution[j];
            }
        }

        // Update population
        for (j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_solution[j];
        }
        opt->population[i].fitness = new_fitness;
    }

    free(new_solution);
    enforce_bound_constraints(opt);
}

// Public interface for population update
void update_population_fsa(Optimizer *opt) {
    // This should not be called directly; handled in FSA_optimize
    fprintf(stderr, "Error: update_population_fsa should be called via FSA_optimize\n");
    exit(1);
}

// Internal helper function for initial strategy update
static void fsa_update_with_initial_strategy(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function) {
    int i, j;
    double new_fitness;
    double *temp_solution = (double *)malloc(opt->dim * sizeof(double));
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));

    // Create temporary population
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            temp_solution[j] = opt->best_solution.position[j] +
                               (opt->best_solution.position[j] - data->local_best_positions[i][j]) * rand_double(0.0, 1.0);
        }

        // Enforce bounds
        for (j = 0; j < opt->dim; j++) {
            if (temp_solution[j] < opt->bounds[2 * j]) {
                temp_solution[j] = opt->bounds[2 * j];
            } else if (temp_solution[j] > opt->bounds[2 * j + 1]) {
                temp_solution[j] = opt->bounds[2 * j + 1];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(temp_solution);
        temp_fitness[i] = new_fitness;

        // Update if better than previous fitness
        if (new_fitness <= data->local_best_values[i]) {
            for (j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = temp_solution[j];
                data->local_best_positions[i][j] = temp_solution[j];
            }
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
        for (j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[best_idx].position[j];
        }
    }

    free(temp_solution);
    free(temp_fitness);
    enforce_bound_constraints(opt);
}

// Public interface for initial strategy update
void update_with_initial_strategy(Optimizer *opt) {
    // This should not be called directly; handled in FSA_optimize
    fprintf(stderr, "Error: update_with_initial_strategy should be called via FSA_optimize\n");
    exit(1);
}

// Main Optimization Function
void FSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int run, iter;
    double *best_scores = (double *)malloc(FSA_NUM_RUNS * sizeof(double));
    double **best_positions = (double **)malloc(FSA_NUM_RUNS * sizeof(double *));
    for (run = 0; run < FSA_NUM_RUNS; run++) {
        best_positions[run] = (double *)malloc(opt->dim * sizeof(double));
    }

    for (run = 0; run < FSA_NUM_RUNS; run++) {
        // Allocate FSAData for this run
        FSAData data;
        fsa_initialize_population(opt, &data, objective_function);

        for (iter = 0; iter < opt->max_iter; iter++) {
            fsa_update_population(opt, &data, objective_function);
            fsa_update_with_initial_strategy(opt, &data, objective_function);
            printf("Run %d, Iteration %d: Best Value = %f\n", run + 1, iter + 1, opt->best_solution.fitness);
        }

        best_scores[run] = opt->best_solution.fitness;
        for (int j = 0; j < opt->dim; j++) {
            best_positions[run][j] = opt->best_solution.position[j];
        }

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
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = best_positions[best_run][j];
    }

    printf("Best Score across all runs: %f\n", opt->best_solution.fitness);
    printf("Best Position: ");
    for (int j = 0; j < opt->dim; j++) {
        printf("%f ", opt->best_solution.position[j]);
    }
    printf("\n");

    // Clean up
    for (run = 0; run < FSA_NUM_RUNS; run++) {
        free(best_positions[run]);
    }
    free(best_positions);
    free(best_scores);
}
