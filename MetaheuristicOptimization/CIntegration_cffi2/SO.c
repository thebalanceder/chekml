#include "SO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>

// Xorshift RNG state
static unsigned long xorshift_state = 0;

// Initialize Xorshift seed
static void init_xorshift(unsigned long seed) {
    xorshift_state = seed ? seed : 1; // Avoid zero seed
}

// Fast Xorshift random number generator
static unsigned long xorshift_rand() {
    unsigned long x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Generate random double between min and max
static double rand_double_so(double min, double max) {
    double t = (double)(xorshift_rand() & 0x7FFFFFFF) / (double)0x7FFFFFFF;
    return min + (max - min) * t;
}

// 🌊 Initialize Population
void initialize_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        if (opt->dim == 2) { // Optimize for common case
            pos[0] = opt->bounds[0] + rand_double_so(0.0, 1.0) * (opt->bounds[1] - opt->bounds[0]);
            pos[1] = opt->bounds[2] + rand_double_so(0.0, 1.0) * (opt->bounds[3] - opt->bounds[2]);
        } else {
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = opt->bounds[2 * j] + rand_double_so(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }
        opt->population[i].fitness = DBL_MAX;
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "initialize_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🌊 Spiral Movement Phase
void spiral_movement_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double r = rand_double_so(0.0, 1.0);
        double theta = 2.0 * M_PI * rand_double_so(0.0, 1.0);
        double ct = cos(theta), st = sin(theta);
        if (opt->dim == 2) { // Optimize for common case
            pos[0] += SPIRAL_STEP * r * ct;
            pos[1] += SPIRAL_STEP * r * st;
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double dir = j < 2 ? (j == 0 ? ct : st) : rand_double_so(-1.0, 1.0);
                pos[j] += SPIRAL_STEP * r * dir;
            }
        }
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "spiral_movement_phase: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🌊 Update and Sort Population
void update_and_sort_population(Optimizer *opt, double (*objective_function)(double *)) {
    // Parallel fitness evaluation
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Update best solution
    double best_fitness = opt->best_solution.fitness;
    int best_idx = -1;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[best_idx].position, sizeof(double) * opt->dim);
    }

    // Sort population by fitness using qsort on indices
    typedef struct {
        int index;
        double fitness;
    } IndexedSolution;

    #define MAX_POP_SIZE 1000
    #define MAX_DIM 100
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

    // Reorder population using index mapping
    double temp_pos[MAX_POP_SIZE * MAX_DIM];
    double temp_fits[MAX_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        temp_fits[i] = opt->population[i].fitness;
        memcpy(temp_pos + i * opt->dim, opt->population[i].position, sizeof(double) * opt->dim);
    }
    for (int i = 0; i < opt->population_size; i++) {
        int old_idx = indexed[i].index;
        opt->population[i].fitness = temp_fits[old_idx];
        memcpy(opt->population[i].position, temp_pos + old_idx * opt->dim, sizeof(double) * opt->dim);
    }
    fprintf(stderr, "update_and_sort_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🚀 Main Optimization Function
void SO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in SO_optimize\n");
        return;
    }
    if (opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid population_size, dim, or max_iter\n");
        return;
    }
    if (opt->population_size > MAX_POP_SIZE || opt->dim > MAX_DIM) {
        fprintf(stderr, "Error: Population size (%d) or dimension (%d) exceeds maximum\n", 
                opt->population_size, opt->dim);
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null position in population[%d]\n", i);
            return;
        }
    }
    init_xorshift((unsigned long)time(NULL));
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
