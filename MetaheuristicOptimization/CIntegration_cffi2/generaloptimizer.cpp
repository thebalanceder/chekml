#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // For parallelization
#include "generaloptimizer.h"
#include "DISOwithRCF.h"
#include "GPC.h"
#include <sycl/sycl.hpp>  // Corrected SYCL header

// Function Prototypes
void enforce_bound_constraints(Optimizer* opt);

Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    if (!opt) {
        fprintf(stderr, "Memory allocation failed for Optimizer\n");
        exit(EXIT_FAILURE);
    }

    opt->dim = dim;
    opt->population_size = population_size;
    opt->max_iter = max_iter;

    opt->bounds = (double*)malloc(2 * dim * sizeof(double));
    if (!opt->bounds) {
        free(opt);
        fprintf(stderr, "Memory allocation failed for bounds\n");
        exit(EXIT_FAILURE);
    }
    memcpy(opt->bounds, bounds, 2 * dim * sizeof(double));

    opt->population = new Solution[population_size];  // Updated to use array instead of vector
    if (!opt->population) {
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "Memory allocation failed for population\n");
        exit(EXIT_FAILURE);
    }

    opt->best_solution.position = (double*)malloc(dim * sizeof(double));
    if (!opt->best_solution.position) {
        free(opt->population);
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "Memory allocation failed for best_solution.position\n");
        exit(EXIT_FAILURE);
    }
    opt->best_solution.fitness = INFINITY;

    // Initialize population within bounds
    double rand_norm = 1.0 / RAND_MAX;
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            double min_bound = bounds[2 * d];
            double max_bound = bounds[2 * d + 1];
            opt->population[i].position[d] = min_bound + ((double)rand() * rand_norm) * (max_bound - min_bound);
        }
        opt->population[i].fitness = INFINITY;
    }

    // Select Optimization Algorithm
    if (strcmp(method, "DISO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DISO_optimize;
    } else if (strcmp(method, "GPC") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GPC_optimize;
    } else {
        fprintf(stderr, "Unknown optimization method: %s\n", method);
        general_free(opt);
        exit(EXIT_FAILURE);
    }

    return opt;
}

// Calls the selected optimization algorithm
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (opt == NULL || opt->optimize == NULL) {
        fprintf(stderr, "Error: Optimizer is not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    
    // Call the assigned optimization method (DISO or GPC)
    opt->optimize(opt, objective_function);
}

// Enforce Bound Constraints for Population and Best Solution
void enforce_bound_constraints(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            }
            if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        if (opt->best_solution.position[j] < opt->bounds[2 * j]) {
            opt->best_solution.position[j] = opt->bounds[2 * j];
        }
        if (opt->best_solution.position[j] > opt->bounds[2 * j + 1]) {
            opt->best_solution.position[j] = opt->bounds[2 * j + 1];
        }
    }
}

// Free Allocated Memory
void general_free(Optimizer* opt) {
    if (!opt) return;
    free(opt->best_solution.position);
    delete[] opt->population;  // Free the array
    free(opt->bounds);
    free(opt);
}

// Retrieve the Best Solution
void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness) {
    if (!opt || !best_position || !best_fitness) return;
    *best_fitness = opt->best_solution.fitness;
    memcpy(best_position, opt->best_solution.position, opt->dim * sizeof(double));
}
