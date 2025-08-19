#include "CS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Inlined random_uniform for better performance
static inline double fast_rand(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline void copy_position(double* dest, const double* src, int dim) {
    for (int d = 0; d < dim; d++) dest[d] = src[d];
}

// Clone the population
static void clone_population(const Optimizer* opt, Solution* clones, int* clone_counts, int clone_size) {
    int total = 0;
    for (int i = 0; i < opt->population_size; i++) {
        int count = clone_size;
        clone_counts[i] = count;
        for (int j = 0; j < count; j++) {
            copy_position(clones[total].position, opt->population[i].position, opt->dim);
            clones[total].fitness = opt->population[i].fitness;
            total++;
        }
    }
}

// Apply hypermutation
static void hypermutation(Solution* clones, int total_clones, int dim, const double* bounds) {
    for (int i = 0; i < total_clones; i++) {
        double* pos = clones[i].position;
        for (int d = 0; d < dim; d++) {
            if (((double)rand() / RAND_MAX) < MUTATION_PROBABILITY) {
                double min = bounds[2 * d];
                double max = bounds[2 * d + 1];
                pos[d] = fast_rand(min, max);
            }
        }
    }
}

// Replace worst individuals
static void replace_worst(Optimizer* opt, int replace_count) {
    int dim = opt->dim;
    for (int i = 0; i < replace_count; i++) {
        int idx = opt->population_size - 1 - i;
        double* pos = opt->population[idx].position;
        for (int d = 0; d < dim; d++) {
            pos[d] = fast_rand(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        opt->population[idx].fitness = INFINITY;
    }
}

// 🚀 Optimized Main
void CS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt) return;

    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;

    // Init population
    for (int i = 0; i < pop_size; i++) {
        double* pos = opt->population[i].position;
        for (int d = 0; d < dim; d++) {
            pos[d] = fast_rand(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        opt->population[i].fitness = objective_function(pos);
    }

    // Init best
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) best_idx = i;
    }
    copy_position(opt->best_solution.position, opt->population[best_idx].position, dim);
    opt->best_solution.fitness = opt->population[best_idx].fitness;

    // Clone space
    int clone_size = (int)(CLONE_FACTOR * pop_size);
    int max_clones = pop_size * clone_size;

    Solution* clones = (Solution*)malloc(sizeof(Solution) * max_clones);
    for (int i = 0; i < max_clones; i++)
        clones[i].position = (double*)malloc(sizeof(double) * dim);

    int* clone_counts = (int*)malloc(sizeof(int) * pop_size);

    // Main loop
    for (int iter = 0; iter < max_iter; iter++) {
        clone_population(opt, clones, clone_counts, clone_size);
        hypermutation(clones, max_clones, dim, opt->bounds);

        for (int i = 0; i < max_clones; i++)
            clones[i].fitness = objective_function(clones[i].position);

        // Select best clone for each original
        int clone_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            Solution* best_clone = &clones[clone_idx];
            for (int j = 1; j < clone_counts[i]; j++) {
                if (clones[clone_idx + j].fitness < best_clone->fitness)
                    best_clone = &clones[clone_idx + j];
            }
            copy_position(opt->population[i].position, best_clone->position, dim);
            opt->population[i].fitness = best_clone->fitness;
            clone_idx += clone_counts[i];
        }

        replace_worst(opt, (int)(REPLACEMENT_RATE * pop_size));

        for (int i = 0; i < pop_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                copy_position(opt->best_solution.position, opt->population[i].position, dim);
                opt->best_solution.fitness = opt->population[i].fitness;
            }
        }

        enforce_bound_constraints(opt);  // optionally inline this too
    }

    for (int i = 0; i < max_clones; i++) free(clones[i].position);
    free(clones);
    free(clone_counts);
}

