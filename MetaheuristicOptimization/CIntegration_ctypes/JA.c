#include "JA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Cruising Phase (Top-performing solutions exploration)
void cruising_phase(Optimizer *opt, int index, int iteration) {
    if (!opt || index < 0 || index >= opt->population_size || !opt->population || !opt->population[index].position) {
        fprintf(stderr, "Invalid parameters in cruising_phase: opt=%p, index=%d, population=%p, position=%p\n",
                (void *)opt, index, (void *)(opt ? opt->population : NULL),
                (void *)(opt && opt->population ? opt->population[index].position : NULL));
        return;
    }

    double *direction = (double *)malloc(opt->dim * sizeof(double));
    if (!direction) {
        fprintf(stderr, "Memory allocation failed for direction in cruising_phase\n");
        return;
    }

    double norm = 0.0;
    // Generate random direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] = rand_double(-1.0, 1.0);
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);
    if (norm == 0.0) norm = 1.0;  // Prevent division by zero

    // Normalize direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] /= norm;
    }

    // Calculate adaptive cruising distance
    double current_cruising_distance = CRUISING_DISTANCE * (1.0 - (double)iteration / opt->max_iter);

    // Update position in-place
    for (int j = 0; j < opt->dim; j++) {
        opt->population[index].position[j] += ALPHA * current_cruising_distance * direction[j];
    }

    free(direction);
    enforce_bound_constraints(opt);
}

// Random Walk Phase (Exploration for non-cruising solutions)
void random_walk_phase_ja(Optimizer *opt, int index) {
    if (!opt || index < 0 || index >= opt->population_size || !opt->population || !opt->population[index].position) {
        fprintf(stderr, "Invalid parameters in random_walk_phase_ja: opt=%p, index=%d, population=%p, position=%p\n",
                (void *)opt, index, (void *)(opt ? opt->population : NULL),
                (void *)(opt && opt->population ? opt->population[index].position : NULL));
        return;
    }

    double *direction = (double *)malloc(opt->dim * sizeof(double));
    if (!direction) {
        fprintf(stderr, "Memory allocation failed for direction in random_walk_phase_ja\n");
        return;
    }

    double norm = 0.0;
    // Generate random direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] = rand_double(-1.0, 1.0);
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);
    if (norm == 0.0) norm = 1.0;  // Prevent division by zero

    // Normalize direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] /= norm;
    }

    // Update position in-place
    for (int j = 0; j < opt->dim; j++) {
        opt->population[index].position[j] += ALPHA * direction[j];
    }

    free(direction);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void JA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || opt->population_size <= 0 || opt->dim <= 0 || !opt->population) {
        fprintf(stderr, "Invalid optimizer parameters: opt=%p, population_size=%d, dim=%d, population=%p\n",
                (void *)opt, opt ? opt->population_size : -1, opt ? opt->dim : -1, (void *)(opt ? opt->population : NULL));
        return;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Store base address for offset verification
    double *base_position = opt->population[0].position;

    // Initialize population fitness
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "population[%d].position is NULL, expected pre-allocated by generaloptimizer.h\n", i);
            return;
        }
        printf("Initial population[%d].position = %p, expected offset = %p\n",
               i, (void *)opt->population[i].position, (void *)(base_position + i * opt->dim));
        assert(opt->population[i].position == base_position + i * opt->dim && "Initial position offset mismatch");
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Initialize best solution
    if (!opt->best_solution.position) {
        fprintf(stderr, "best_solution.position is NULL, expected pre-allocated by generaloptimizer.h\n");
        return;
    }
    printf("Initial best_solution.position = %p\n", (void *)opt->best_solution.position);
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Allocate index array for sorting
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed for indices in JA_optimize\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            if (!opt->population[i].position) {
                fprintf(stderr, "Null position at population[%d] during iteration %d\n", i, iter);
                free(indices);
                return;
            }
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }

        // Sort indices by fitness
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (opt->population[indices[j]].fitness > opt->population[indices[j + 1]].fitness) {
                    int temp = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp;
                }
            }
        }

        // Verify population pointers remain unchanged
        for (int i = 0; i < opt->population_size; i++) {
            /*printf("Post-sorting iteration %d: population[%d].position = %p, expected offset = %p\n",
                   iter, i, (void *)opt->population[i].position, (void *)(base_position + i * opt->dim));*/
            assert(opt->population[i].position == base_position + i * opt->dim && "Position offset mismatch after sorting");
        }

        // Update best solution
        if (opt->population[indices[0]].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[indices[0]].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[indices[0]].position[j];
            }
        }

        // Determine number of cruising individuals
        int num_cruising = (int)(CRUISING_PROBABILITY * opt->population_size);
        if (num_cruising < 0 || num_cruising > opt->population_size) {
            fprintf(stderr, "Invalid num_cruising: %d\n", num_cruising);
            free(indices);
            return;
        }

        // Update cruising individuals
        for (int i = 0; i < num_cruising; i++) {
            cruising_phase(opt, indices[i], iter);
            if (opt->population[indices[i]].position) {
                opt->population[indices[i]].fitness = objective_function(opt->population[indices[i]].position);
            }
        }

        // Update remaining individuals with random walk
        for (int i = num_cruising; i < opt->population_size; i++) {
            random_walk_phase_ja(opt, indices[i]);
            if (opt->population[indices[i]].position) {
                opt->population[indices[i]].fitness = objective_function(opt->population[indices[i]].position);
            }
        }

        // Update best solution after phase updates
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Log final population pointers
    for (int i = 0; i < opt->population_size; i++) {
        printf("Final population[%d].position = %p, expected offset = %p\n",
               i, (void *)opt->population[i].position, (void *)(base_position + i * opt->dim));
        assert(opt->population[i].position == base_position + i * opt->dim && "Final position offset mismatch");
    }

    // Free index array
    free(indices);
}
