#include "CS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// 🎲 Generate a random double between min and max
static double random_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// 📌 Clone the population based on fitness
static void clone_population(Optimizer* opt, Solution* clones, int* clone_counts) {
    int total_clones = 0;
    for (int i = 0; i < opt->population_size; i++) {
        int count = (int)(CLONE_FACTOR * opt->population_size);
        clone_counts[i] = count;
        for (int j = 0; j < count; j++) {
            memcpy(clones[total_clones].position, opt->population[i].position, sizeof(double) * opt->dim);
            clones[total_clones].fitness = opt->population[i].fitness;
            total_clones++;
        }
    }
}

// 🔄 Apply hypermutation to clones
static void hypermutation(Solution* clones, int total_clones, int dim, double* bounds) {
    for (int i = 0; i < total_clones; i++) {
        for (int d = 0; d < dim; d++) {
            if (random_uniform(0.0, 1.0) < MUTATION_PROBABILITY) {
                double min_bound = bounds[2 * d];
                double max_bound = bounds[2 * d + 1];
                clones[i].position[d] = random_uniform(min_bound, max_bound);
            }
        }
    }
}

// 🔄 Replace worst individuals with new random ones
static void replace_worst(Optimizer* opt, int count) {
    for (int i = 0; i < count; i++) {
        int idx = opt->population_size - 1 - i;
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[idx].position[d] = random_uniform(min_bound, max_bound);
        }
        opt->population[idx].fitness = INFINITY;
    }
}

// 🚀 Main Optimization Function
void CS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt) return;

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = random_uniform(min_bound, max_bound);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Find initial best solution
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    memcpy(opt->best_solution.position, opt->population[best_idx].position, sizeof(double) * opt->dim);
    opt->best_solution.fitness = opt->population[best_idx].fitness;

    // Allocate memory for clones
    int max_clones = opt->population_size * (int)(CLONE_FACTOR * opt->population_size);
    Solution* clones = (Solution*)malloc(sizeof(Solution) * max_clones);
    for (int i = 0; i < max_clones; i++) {
        clones[i].position = (double*)malloc(sizeof(double) * opt->dim);
    }
    int* clone_counts = (int*)malloc(sizeof(int) * opt->population_size);

    // Optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Clone population
        clone_population(opt, clones, clone_counts);

        // Hypermutation
        hypermutation(clones, max_clones, opt->dim, opt->bounds);

        // Evaluate clones
        for (int i = 0; i < max_clones; i++) {
            clones[i].fitness = objective_function(clones[i].position);
        }

        // Select best clones to replace population
        int clone_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            Solution* best_clone = &clones[clone_idx];
            for (int j = 1; j < clone_counts[i]; j++) {
                if (clones[clone_idx + j].fitness < best_clone->fitness) {
                    best_clone = &clones[clone_idx + j];
                }
            }
            memcpy(opt->population[i].position, best_clone->position, sizeof(double) * opt->dim);
            opt->population[i].fitness = best_clone->fitness;
            clone_idx += clone_counts[i];
        }

        // Replace worst individuals
        int replace_count = (int)(REPLACEMENT_RATE * opt->population_size);
        replace_worst(opt, replace_count);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
                opt->best_solution.fitness = opt->population[i].fitness;
            }
        }

        enforce_bound_constraints(opt);
    }

    // Free allocated memory
    for (int i = 0; i < max_clones; i++) {
        free(clones[i].position);
    }
    free(clones);
    free(clone_counts);
}