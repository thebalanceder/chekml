#include "MA.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static double rand_normal(double stddev) {
    // Box-Muller transform
    double u1 = rand_uniform(0.0, 1.0);
    double u2 = rand_uniform(0.0, 1.0);
    return stddev * sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

static void clamp_solution(double* solution, int dim, const double* bounds) {
    for (int i = 0; i < dim; i++) {
        if (solution[i] < bounds[2 * i]) solution[i] = bounds[2 * i];
        if (solution[i] > bounds[2 * i + 1]) solution[i] = bounds[2 * i + 1];
    }
}

static void hill_climbing(double* solution, int dim, const double* bounds, ObjectiveFunction obj_func) {
    double current_fitness = obj_func(solution);
    double* new_solution = (double*)malloc(dim * sizeof(double));

    for (int it = 0; it < MA_LOCAL_SEARCH_ITERS; it++) {
        for (int j = 0; j < dim; j++) {
            new_solution[j] = solution[j] + rand_normal(0.1);
        }
        clamp_solution(new_solution, dim, bounds);

        double new_fitness = obj_func(new_solution);
        if (new_fitness < current_fitness) {
            memcpy(solution, new_solution, dim * sizeof(double));
            current_fitness = new_fitness;
        }
    }

    free(new_solution);
}

static void crossover(const double* parent1, const double* parent2, double* child1, double* child2, int dim) {
    int cp = rand() % dim;
    for (int i = 0; i < dim; i++) {
        if (i <= cp) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        } else {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
}

static void mutate(double* solution, int dim, const double* bounds) {
    for (int i = 0; i < dim; i++) {
        solution[i] += rand_normal(MA_MUTATION_STDDEV);
    }
    clamp_solution(solution, dim, bounds);
}

void MA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;

    double** population = (double**)malloc(pop_size * sizeof(double*));
    double* fitness = (double*)malloc(pop_size * sizeof(double));
    for (int i = 0; i < pop_size; i++) {
        population[i] = (double*)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            population[i][j] = rand_uniform(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        fitness[i] = objective_function(population[i]);
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Hill climbing on top individuals
        int ls_size = (int)(MA_LOCAL_SEARCH_RATE * pop_size);
        for (int i = 0; i < ls_size; i++) {
            int idx = rand() % pop_size;
            hill_climbing(population[idx], dim, opt->bounds, objective_function);
            fitness[idx] = objective_function(population[idx]);
        }

        // Sort by fitness
        int* indices = (int*)malloc(pop_size * sizeof(int));
        for (int i = 0; i < pop_size; i++) indices[i] = i;
        for (int i = 0; i < pop_size - 1; i++) {
            for (int j = i + 1; j < pop_size; j++) {
                if (fitness[indices[i]] > fitness[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        int num_parents = (int)(MA_CROSSOVER_RATE * pop_size);
        if (num_parents % 2 != 0) num_parents--;  // ensure even number

        // Generate children
        double** children = (double**)malloc(num_parents * sizeof(double*));
        for (int i = 0; i < num_parents; i += 2) {
            int p1 = indices[i];
            int p2 = indices[i + 1];
            children[i] = (double*)malloc(dim * sizeof(double));
            children[i + 1] = (double*)malloc(dim * sizeof(double));
            crossover(population[p1], population[p2], children[i], children[i + 1], dim);

            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate(children[i], dim, opt->bounds);
            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate(children[i + 1], dim, opt->bounds);
        }

        // Replace worst individuals
        for (int i = 0; i < num_parents; i++) {
            int idx = indices[pop_size - 1 - i];
            memcpy(population[idx], children[i], dim * sizeof(double));
            fitness[idx] = objective_function(population[idx]);
            free(children[i]);
        }

        free(children);
        free(indices);

        // Track best
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < fitness[best_idx]) best_idx = i;
        }

        if (fitness[best_idx] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[best_idx];
            memcpy(opt->best_solution.position, population[best_idx], dim * sizeof(double));
        }

        printf("MA | Iteration %4d -> Best Fitness = %0.16f\n", iter + 1, opt->best_solution.fitness);
    }

    for (int i = 0; i < pop_size; i++) free(population[i]);
    free(population);
    free(fitness);
}

