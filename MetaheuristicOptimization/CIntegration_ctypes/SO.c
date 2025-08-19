#include "SO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

// Function to generate a random double between min and max
double rand_double_so(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🌊 Initialize Population
void initialize_population(Optimizer *opt) {
    if (!opt || !opt->population || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in initialize_population\n");
        return;
    }
    if (opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Error: Invalid population_size or dim\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null position in population[%d]\n", i);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double_so(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = DBL_MAX;
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "initialize_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🌊 Spiral Movement Phase
void spiral_movement_phase(Optimizer *opt) {
    if (!opt || !opt->population) {
        fprintf(stderr, "Error: Null pointer in spiral_movement_phase\n");
        return;
    }
    if (opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Error: Invalid population_size or dim\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null position in population[%d]\n", i);
            return;
        }
        double r = rand_double_so(0.0, 1.0);
        double theta = 2.0 * M_PI * rand_double_so(0.0, 1.0);
        double direction[2];
        direction[0] = cos(theta);
        direction[1] = sin(theta);

        for (int j = 0; j < opt->dim; j++) {
            double step = SPIRAL_STEP * r * (j < 2 ? direction[j] : rand_double_so(-1.0, 1.0));
            opt->population[i].position[j] += step;
        }
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "spiral_movement_phase: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🌊 Update and Sort Population
void update_and_sort_population(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !objective_function || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in update_and_sort_population\n");
        return;
    }
    if (opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Error: Invalid population_size or dim\n");
        return;
    }
    // Evaluate fitness and update best solution
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null position in population[%d]\n", i);
            return;
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
        }
    }

    // Sort population by fitness using qsort on indices
    typedef struct {
        int index;
        double fitness;
    } IndexedSolution;

    #define MAX_POP_SIZE 1000
    #define MAX_DIM 100
    if (opt->population_size > MAX_POP_SIZE) {
        fprintf(stderr, "Error: Population size exceeds maximum (%d)\n", MAX_POP_SIZE);
        return;
    }
    if (opt->dim > MAX_DIM) {
        fprintf(stderr, "Error: Dimension exceeds maximum (%d)\n", MAX_DIM);
        return;
    }
    IndexedSolution indexed[MAX_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        indexed[i].index = i;
        indexed[i].fitness = opt->population[i].fitness;
    }

    int compare_by_fitness(const void *a, const void *b) {
        const IndexedSolution *ia = (const IndexedSolution *)a;
        const IndexedSolution *ib = (const IndexedSolution *)b;
        return (ia->fitness > ib->fitness) - (ia->fitness < ib->fitness);
    }
    qsort(indexed, opt->population_size, sizeof(IndexedSolution), compare_by_fitness);

    // Reorder population by swapping position and fitness data
    double temp_pos[MAX_POP_SIZE * MAX_DIM];
    double temp_fit;
    for (int i = 0; i < opt->population_size; i++) {
        int new_idx = indexed[i].index;
        if (new_idx != i) {
            // Swap position
            memcpy(temp_pos, opt->population[i].position, sizeof(double) * opt->dim);
            memcpy(opt->population[i].position, opt->population[new_idx].position, sizeof(double) * opt->dim);
            memcpy(opt->population[new_idx].position, temp_pos, sizeof(double) * opt->dim);
            // Swap fitness
            temp_fit = opt->population[i].fitness;
            opt->population[i].fitness = opt->population[new_idx].fitness;
            opt->population[new_idx].fitness = temp_fit;
        }
    }
    fprintf(stderr, "update_and_sort_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🚀 Main Optimization Function
void SO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Null pointer in SO_optimize\n");
        return;
    }
    if (opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid max_iter\n");
        return;
    }
    fprintf(stderr, "SO_optimize: starting, population[0].position=%p\n", (void*)opt->population[0].position);
    initialize_population(opt);
    for (int iter = 0; iter < opt->max_iter; iter++) {
        spiral_movement_phase(opt);
        update_and_sort_population(opt, objective_function);
        enforce_bound_constraints(opt);
    }
    fprintf(stderr, "SO_optimize: completed, population[0].position=%p, best_solution.position=%p\n", 
            (void*)opt->population[0].position, (void*)opt->best_solution.position);
}
