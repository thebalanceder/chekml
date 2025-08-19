#include "JA.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <time.h>

// Fast thread-safe random number generation
static inline double rand_double(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

// Generate random direction normalized
static void generate_random_direction(double *direction, int dim, unsigned int *seed) {
    double norm = 0.0;
    for (int j = 0; j < dim; j++) {
        direction[j] = -1.0 + 2.0 * rand_double(seed);
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);
    if (norm == 0.0) norm = 1.0;
    #pragma omp simd
    for (int j = 0; j < dim; j++) {
        direction[j] /= norm;
    }
}

// Cruising phase: exploitation
static void ja_cruising(Optimizer *opt, int idx, int iteration, double *direction, unsigned int *seed) {
    double *pos = opt->population[idx].position;
    generate_random_direction(direction, opt->dim, seed);

    double distance = JA_CRUISING_DISTANCE * (1.0 - (double)iteration / opt->max_iter);
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        pos[j] += JA_ALPHA * distance * direction[j];
    }
}

// Random walk phase: exploration
static void ja_random_walk(Optimizer *opt, int idx, double *direction, unsigned int *seed) {
    double *pos = opt->population[idx].position;
    generate_random_direction(direction, opt->dim, seed);

    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        pos[j] += JA_ALPHA * direction[j];
    }
}

// Fast comparison for sorting
static int compare_fitness(const void *a, const void *b) {
    const Solution *sa = *(const Solution **)a;
    const Solution *sb = *(const Solution **)b;
    return (sa->fitness > sb->fitness) - (sa->fitness < sb->fitness);
}

// Enforce boundaries and re-evaluate individuals
static void enforce_and_evaluate(Optimizer *opt, double (*objective_function)(double *)) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        enforce_bound_constraints(opt);
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Main Jaguar Algorithm
void JA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Invalid optimizer parameters\n");
        return;
    }

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int max_iter = opt->max_iter;
    int num_cruising = (int)(JA_CRUISING_PROBABILITY * pop_size);

    unsigned int *seeds = malloc(sizeof(unsigned int) * omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); i++) {
        seeds[i] = (unsigned int)time(NULL) ^ (i * 1234567);
    }

    double **directions = malloc(sizeof(double *) * omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); i++) {
        directions[i] = malloc(sizeof(double) * dim);
    }

    enforce_and_evaluate(opt, objective_function);

    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
        }
    }

    Solution **sorted = malloc(sizeof(Solution *) * pop_size);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            sorted[i] = &opt->population[i];
        }

        qsort(sorted, pop_size, sizeof(Solution *), compare_fitness);

        if (sorted[0]->fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = sorted[0]->fitness;
            memcpy(opt->best_solution.position, sorted[0]->position, sizeof(double) * dim);
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *dir = directions[tid];
            unsigned int seed = seeds[tid];

            #pragma omp for
            for (int i = 0; i < pop_size; i++) {
                int idx = (int)(sorted[i] - opt->population);  // get original index
                if (i < num_cruising) {
                    ja_cruising(opt, idx, iter, dir, &seed);
                } else {
                    ja_random_walk(opt, idx, dir, &seed);
                }
            }
            seeds[tid] = seed;
        }

        enforce_and_evaluate(opt, objective_function);
    }

    for (int i = 0; i < omp_get_max_threads(); i++) {
        free(directions[i]);
    }
    free(directions);
    free(seeds);
    free(sorted);
}

