#include "FirefA.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

// Fast random double generator using linear congruential generator
static inline double fast_rand(double min, double max) {
    static unsigned long seed = 1;
    seed = (seed * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL;
    double r = (double)seed / 0xFFFFFFFFUL;
    return min + (max - min) * r;
}

// Initialize firefly population
void initialize_fireflies(Optimizer *opt, ObjectiveFunction objective_function, double **positions_out, int minimize) {
    // Precompute bounds for faster access
    double *lb = (double*)malloc(opt->dim * sizeof(double));
    double *ub = (double*)malloc(opt->dim * sizeof(double));
    if (!lb || !ub) {
        fprintf(stderr, "Memory allocation failed for bounds\n");
        free(lb); free(ub);
        exit(1);
    }
    for (int j = 0; j < opt->dim; j++) {
        lb[j] = opt->bounds[2 * j];
        ub[j] = opt->bounds[2 * j + 1];
    }

    // Allocate all positions and fitness values at once
    double *positions = (double*)malloc(opt->population_size * opt->dim * sizeof(double));
    double *fitness = (double*)malloc(opt->population_size * sizeof(double));
    if (!positions || !fitness) {
        fprintf(stderr, "Memory allocation failed for positions or fitness\n");
        free(lb); free(ub); free(positions); free(fitness);
        exit(1);
    }

    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].position = positions + i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = lb[j] + (ub[j] - lb[j]) * fast_rand(0.0, 1.0);
        }
        fitness[i] = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness[i];
    }

    // Initialize best solution
    opt->best_solution.position = (double*)malloc(opt->dim * sizeof(double));
    if (!opt->best_solution.position) {
        fprintf(stderr, "Memory allocation failed for best solution\n");
        free(lb); free(ub); free(positions); free(fitness);
        exit(1);
    }
    int best_idx = 0;
    opt->best_solution.fitness = fitness[0];
    for (int i = 1; i < opt->population_size; i++) {
        if ((minimize && fitness[i] < opt->best_solution.fitness) ||
            (!minimize && fitness[i] > opt->best_solution.fitness)) {
            opt->best_solution.fitness = fitness[i];
            best_idx = i;
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    // Verify initial best solution
    double verify_fitness = objective_function(opt->best_solution.position);
    if (fabs(verify_fitness - opt->best_solution.fitness) > 1e-6) {
        fprintf(stderr, "Warning: Initial best solution fitness mismatch: stored=%f, computed=%f\n",
                opt->best_solution.fitness, verify_fitness);
    }

    enforce_bound_constraints(opt);
    free(lb);
    free(ub);
    free(fitness);
    *positions_out = positions; // Return the positions pointer for later freeing
}

// Fast approximation of exp(-x) for small x
static inline double fast_exp_neg(double x) {
    if (x < 0.01) return 1.0 - x; // Linear approximation for small x
    return exp(-x);
}

// Update firefly positions
void update_fireflies(Optimizer *opt, int t, ObjectiveFunction objective_function, int minimize) {
    double alpha = FA_ALPHA * pow(FA_THETA, t);
    double *scale = (double*)malloc(opt->dim * sizeof(double));
    if (!scale) {
        fprintf(stderr, "Memory allocation failed for scale\n");
        exit(1);
    }
    for (int j = 0; j < opt->dim; j++) {
        scale[j] = fabs(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
    }

    // Precompute fitness to avoid redundant calls
    double *fitness = (double*)malloc(opt->population_size * sizeof(double));
    if (!fitness) {
        fprintf(stderr, "Memory allocation failed for fitness\n");
        free(scale);
        exit(1);
    }
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness[i];
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->population_size; j++) {
            if ((minimize && fitness[i] > fitness[j]) || // Move if j is brighter (lower fitness for minimization)
                (!minimize && fitness[i] < fitness[j])) { // Move if j is brighter (higher fitness for maximization)
                double r = 0.0;
                double *pos_i = opt->population[i].position;
                double *pos_j = opt->population[j].position;
                for (int k = 0; k < opt->dim; k++) {
                    double diff = pos_i[k] - pos_j[k];
                    r += diff * diff;
                }
                r = sqrt(r);
                double beta = FA_BETA0 * fast_exp_neg(FA_GAMMA * r * r);
                for (int k = 0; k < opt->dim; k++) {
                    double step = alpha * (fast_rand(0.0, 1.0) - 0.5) * scale[k];
                    pos_i[k] += beta * (pos_j[k] - pos_i[k]) + step;
                }
            }
        }
    }

    enforce_bound_constraints(opt);
    free(scale);
    free(fitness);
}

// Rank fireflies using insertion sort
void rank_fireflies(Optimizer *opt, int minimize) {
    for (int i = 1; i < opt->population_size; i++) {
        double key_fitness = opt->population[i].fitness;
        double *key_pos = opt->population[i].position;
        int j = i - 1;
        while (j >= 0 && ((minimize && opt->population[j].fitness > key_fitness) ||
                          (!minimize && opt->population[j].fitness < key_fitness))) {
            opt->population[j + 1].fitness = opt->population[j].fitness;
            opt->population[j + 1].position = opt->population[j].position;
            j--;
        }
        opt->population[j + 1].fitness = key_fitness;
        opt->population[j + 1].position = key_pos;
    }
}

// Main optimization function
void FirefA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    // Assume minimization by default; change to 0 for maximization if needed
    int minimize = 1;
    double *positions = NULL; // To store the original positions pointer
    initialize_fireflies(opt, objective_function, &positions, minimize);

    for (int t = 0; t < opt->max_iter; t++) {
        update_fireflies(opt, t, objective_function, minimize);
        rank_fireflies(opt, minimize);

        // Update best solution
        if ((minimize && opt->population[0].fitness < opt->best_solution.fitness) ||
            (!minimize && opt->population[0].fitness > opt->best_solution.fitness)) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
            // Verify best solution
            double verify_fitness = objective_function(opt->best_solution.position);
            if (fabs(verify_fitness - opt->best_solution.fitness) > 1e-6) {
                fprintf(stderr, "Warning: Best solution fitness mismatch at iteration %d: stored=%f, computed=%f\n",
                        t + 1, opt->best_solution.fitness, verify_fitness);
            }
        }

        // Log progress for debugging
        printf("Iteration %d: Best Fitness = %f, Position = [", t + 1, opt->best_solution.fitness);
        for (int j = 0; j < opt->dim; j++) {
            printf("%f", opt->best_solution.position[j]);
            if (j < opt->dim - 1) printf(", ");
        }
        printf("]\n");
    }

    // Log final best solution
    printf("Final Best Fitness = %f, Position = [", opt->best_solution.fitness);
    for (int j = 0; j < opt->dim; j++) {
        printf("%f", opt->best_solution.position[j]);
        if (j < opt->dim - 1) printf(", ");
    }
    printf("]\n");

    // Free allocated memory (except best_solution.position, to be freed by caller)
    free(positions); // Free the contiguous position array
    // Do not free opt->best_solution.position to allow Python to access it
}
