#include "JA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

// Structure to hold context for qsort
typedef struct {
    Optimizer *opt;
    int *indices;
} SortContext;

// Global context for qsort (for portability)
static SortContext global_sort_ctx;

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    double fitness_a = global_sort_ctx.opt->population[idx_a].fitness;
    double fitness_b = global_sort_ctx.opt->population[idx_b].fitness;
    return (fitness_a > fitness_b) - (fitness_a < fitness_b);
}

// Function to generate a random double between min and max
static inline double rand_double_ja(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Cruising Phase (Top-performing solutions exploration)
void ja_cruising_phase(Optimizer *opt, int index, int iteration, double *direction) {
    if (!opt || index < 0 || index >= opt->population_size || !opt->population || !opt->population[index].position) {
        fprintf(stderr, "Invalid parameters in ja_cruising_phase: opt=%p, index=%d, population=%p, position=%p\n",
                (void *)opt, index, (void *)(opt ? opt->population : NULL),
                (void *)(opt && opt->population ? opt->population[index].position : NULL));
        return;
    }

    double norm = 0.0;
    // Generate random direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] = rand_double_ja(-1.0, 1.0);
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

    enforce_bound_constraints(opt);
}

// Random Walk Phase (Exploration for non-cruising solutions)
void ja_random_walk_phase(Optimizer *opt, int index, double *direction) {
    if (!opt || index < 0 || index >= opt->population_size || !opt->population || !opt->population[index].position) {
        fprintf(stderr, "Invalid parameters in ja_random_walk_phase: opt=%p, index=%d, population=%p, position=%p\n",
                (void *)opt, index, (void *)(opt ? opt->population : NULL),
                (void *)(opt && opt->population ? opt->population[index].position : NULL));
        return;
    }

    double norm = 0.0;
    // Generate random direction
    for (int j = 0; j < opt->dim; j++) {
        direction[j] = rand_double_ja(-1.0, 1.0);
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

    // Allocate reusable direction array
    double *direction = (double *)malloc(opt->dim * sizeof(double));
    if (!direction) {
        fprintf(stderr, "Memory allocation failed for direction in JA_optimize\n");
        return;
    }

    // Allocate index array for sorting
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed for indices in JA_optimize\n");
        free(direction);
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Allocate flags to track which individuals need fitness updates
    char *needs_fitness_update = (char *)malloc(opt->population_size * sizeof(char));
    if (!needs_fitness_update) {
        fprintf(stderr, "Memory allocation failed for needs_fitness_update in JA_optimize\n");
        free(indices);
        free(direction);
        return;
    }
    memset(needs_fitness_update, 1, opt->population_size * sizeof(char)); // All need initial update

    // Initialize population fitness in parallel
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "population[%d].position is NULL, expected pre-allocated by generaloptimizer.h\n", i);
            continue;
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Initialize best solution
    if (!opt->best_solution.position) {
        fprintf(stderr, "best_solution.position is NULL, expected pre-allocated by generaloptimizer.h\n");
        free(needs_fitness_update);
        free(indices);
        free(direction);
        return;
    }
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            #pragma omp critical
            {
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update fitness only for modified individuals
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            if (needs_fitness_update[i] && opt->population[i].position) {
                opt->population[i].fitness = objective_function(opt->population[i].position);
                needs_fitness_update[i] = 0;
            }
        }

        // Sort indices using qsort
        global_sort_ctx.opt = opt;
        global_sort_ctx.indices = indices;
        qsort(indices, opt->population_size, sizeof(int), compare_fitness);

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
            free(needs_fitness_update);
            free(indices);
            free(direction);
            return;
        }

        // Update cruising and random walk individuals in parallel
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            if (i < num_cruising) {
                ja_cruising_phase(opt, indices[i], iter, direction);
            } else {
                ja_random_walk_phase(opt, indices[i], direction);
            }
            needs_fitness_update[indices[i]] = 1; // Mark for fitness update
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
    }

    // Clean up
    free(needs_fitness_update);
    free(indices);
    free(direction);
}
