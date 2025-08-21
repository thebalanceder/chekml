#include "MA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline double rand_normal(double stddev) {
    double u1 = rand_uniform(0.0, 1.0);
    double u2 = rand_uniform(0.0, 1.0);
    return stddev * sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

static inline void clamp(double* vec, int dim, const double* bounds) {
    for (int i = 0; i < dim; i++) {
        vec[i] = MAX(bounds[2 * i], MIN(bounds[2 * i + 1], vec[i]));
    }
}

static void hill_climb(double* sol, int dim, const double* bounds, ObjectiveFunction f) {
    double* candidate = (double*)malloc(dim * sizeof(double));
    double best_fit = f(sol);

    for (int it = 0; it < MA_LOCAL_SEARCH_ITERS; it++) {
        for (int j = 0; j < dim; j++)
            candidate[j] = sol[j] + rand_normal(MA_MUTATION_STDDEV);
        clamp(candidate, dim, bounds);

        double fit = f(candidate);
        if (fit < best_fit) {
            memcpy(sol, candidate, dim * sizeof(double));
            best_fit = fit;
        }
    }
    free(candidate);
}

static void crossover(const double* p1, const double* p2, double* c1, double* c2, int dim) {
    int cx = rand() % dim;
    for (int i = 0; i < dim; i++) {
        c1[i] = (i <= cx) ? p1[i] : p2[i];
        c2[i] = (i <= cx) ? p2[i] : p1[i];
    }
}

static void mutate(double* sol, int dim, const double* bounds) {
    for (int i = 0; i < dim; i++)
        sol[i] += rand_normal(MA_MUTATION_STDDEV);
    clamp(sol, dim, bounds);
}

void MA_optimize(Optimizer* opt, ObjectiveFunction f) {
    int dim = opt->dim, pop = opt->population_size, max_iter = opt->max_iter;
    int total = pop * dim;
    double* population = (double*)malloc(sizeof(double) * total);
    double* fitness = (double*)malloc(sizeof(double) * pop);

    // Bounds copy
    double* bounds = (double*)malloc(sizeof(double) * dim * 2);
    memcpy(bounds, opt->bounds, sizeof(double) * dim * 2);

    // Initial population
    #pragma omp parallel for
    for (int i = 0; i < pop; i++) {
        for (int d = 0; d < dim; d++)
            population[i * dim + d] = rand_uniform(bounds[2 * d], bounds[2 * d + 1]);
        fitness[i] = f(&population[i * dim]);
    }

    double* buffer = (double*)malloc(sizeof(double) * dim);
    double* children = (double*)malloc(sizeof(double) * pop * dim);  // worst-case for space reuse

    for (int iter = 0; iter < max_iter; iter++) {
        int ls_count = (int)(MA_LOCAL_SEARCH_RATE * pop);

        // Hill Climbing in Parallel
        #pragma omp parallel for private(buffer)
        for (int i = 0; i < ls_count; i++) {
            int idx = rand() % pop;
            hill_climb(&population[idx * dim], dim, bounds, f);
            fitness[idx] = f(&population[idx * dim]);
        }

        // Select top individuals by sorting indices
        int* indices = (int*)malloc(sizeof(int) * pop);
        for (int i = 0; i < pop; i++) indices[i] = i;
        for (int i = 0; i < pop - 1; i++) {
            for (int j = i + 1; j < pop; j++) {
                if (fitness[indices[i]] > fitness[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        // Crossover and Mutation
        int num_children = ((int)(MA_CROSSOVER_RATE * pop)) & ~1;  // even
        for (int i = 0; i < num_children; i += 2) {
            int p1 = indices[i], p2 = indices[i + 1];
            double* c1 = &children[i * dim];
            double* c2 = &children[(i + 1) * dim];
            crossover(&population[p1 * dim], &population[p2 * dim], c1, c2, dim);
            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate(c1, dim, bounds);
            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate(c2, dim, bounds);
        }

        // Replace worst with children
        for (int i = 0; i < num_children; i++) {
            int idx = indices[pop - 1 - i];
            memcpy(&population[idx * dim], &children[i * dim], sizeof(double) * dim);
            fitness[idx] = f(&population[idx * dim]);
        }

        // Best solution tracking
        int best_idx = 0;
        for (int i = 1; i < pop; i++)
            if (fitness[i] < fitness[best_idx]) best_idx = i;

        if (fitness[best_idx] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[best_idx];
            memcpy(opt->best_solution.position, &population[best_idx * dim], dim * sizeof(double));
        }

        printf("MA | Iteration %4d -> Best Fitness = %.16f\n", iter + 1, opt->best_solution.fitness);
        free(indices);
    }

    free(population);
    free(fitness);
    free(bounds);
    free(children);
}

