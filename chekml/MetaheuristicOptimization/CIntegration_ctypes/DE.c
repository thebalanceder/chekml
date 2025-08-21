#include "DE.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

static double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static void mutate(double* mutant, double** pop, double* best, int i, int pop_size, int dim, int strategy) {
    int r1, r2, r3;
    do { r1 = rand() % pop_size; } while (r1 == i);
    do { r2 = rand() % pop_size; } while (r2 == i || r2 == r1);
    do { r3 = rand() % pop_size; } while (r3 == i || r3 == r1 || r3 == r2);

    switch (strategy) {
        case STRATEGY_RAND_1_BIN:
            for (int d = 0; d < dim; d++)
                mutant[d] = pop[r1][d] + DE_F * (pop[r2][d] - pop[r3][d]);
            break;
        case STRATEGY_BEST_1_BIN:
            for (int d = 0; d < dim; d++)
                mutant[d] = best[d] + DE_F * (pop[r1][d] - pop[r2][d]);
            break;
        case STRATEGY_RAND_TO_BEST_1:
            for (int d = 0; d < dim; d++)
                mutant[d] = pop[r1][d] + 0.5 * (best[d] - pop[r1][d]) + DE_F * (pop[r2][d] - pop[r3][d]);
            break;
    }
}

static void crossover(double* trial, double* target, double* mutant, int dim) {
    for (int d = 0; d < dim; d++) {
        if (((double)rand() / RAND_MAX) < DE_CR || d == rand() % dim)
            trial[d] = mutant[d];
        else
            trial[d] = target[d];
    }
}

void DE_optimize(Optimizer* opt, ObjectiveFunction objective_function, int strategy) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;

    double** pop = (double**)malloc(sizeof(double*) * pop_size);
    double* fitness = (double*)malloc(sizeof(double) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i] = (double*)malloc(sizeof(double) * dim);
        for (int d = 0; d < dim; d++) {
            pop[i][d] = rand_uniform(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        fitness[i] = objective_function(pop[i]);
    }

    double* best = (double*)malloc(sizeof(double) * dim);
    double best_fit = fitness[0];
    memcpy(best, pop[0], sizeof(double) * dim);
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fit) {
            best_fit = fitness[i];
            memcpy(best, pop[i], sizeof(double) * dim);
        }
    }

    double* mutant = (double*)malloc(sizeof(double) * dim);
    double* trial = (double*)malloc(sizeof(double) * dim);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            mutate(mutant, pop, best, i, pop_size, dim, strategy=STRATEGY);
            crossover(trial, pop[i], mutant, dim);

            for (int d = 0; d < dim; d++) {
                if (trial[d] < opt->bounds[2 * d]) trial[d] = opt->bounds[2 * d];
                if (trial[d] > opt->bounds[2 * d + 1]) trial[d] = opt->bounds[2 * d + 1];
            }

            double trial_fit = objective_function(trial);
            if (trial_fit < fitness[i]) {
                memcpy(pop[i], trial, sizeof(double) * dim);
                fitness[i] = trial_fit;
                if (trial_fit < best_fit) {
                    best_fit = trial_fit;
                    memcpy(best, trial, sizeof(double) * dim);
                }
            }
        }
        printf("DE|%5d -----> %9.16f\n", iter + 1, best_fit);
    }

    memcpy(opt->best_solution.position, best, sizeof(double) * dim);
    opt->best_solution.fitness = best_fit;

    for (int i = 0; i < pop_size; i++) free(pop[i]);
    free(pop);
    free(fitness);
    free(best);
    free(mutant);
    free(trial);
}