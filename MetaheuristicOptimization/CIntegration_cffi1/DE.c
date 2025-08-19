#include "DE.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stddef.h>

#define CACHE_LINE 64
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define rand_int(max) (rand() % (max))

static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline void mutate_and_crossover(
    double *restrict trial, const double *restrict target,
    double *restrict pop, const double *restrict best,
    int i, int pop_size, int dim, int strategy)
{
    int r1, r2, r3;
    do { r1 = rand_int(pop_size); } while (r1 == i);
    do { r2 = rand_int(pop_size); } while (r2 == i || r2 == r1);
    do { r3 = rand_int(pop_size); } while (r3 == i || r3 == r1 || r3 == r2);

    int j_rand = rand_int(dim);
    int base_i = i * dim;
    int base_r1 = r1 * dim;
    int base_r2 = r2 * dim;
    int base_r3 = r3 * dim;

    for (int d = 0; d < dim; d++) {
        double v;
        switch (strategy) {
            case STRATEGY_RAND_1_BIN:
                v = pop[base_r1 + d] + DE_F * (pop[base_r2 + d] - pop[base_r3 + d]);
                break;
            case STRATEGY_BEST_1_BIN:
                v = best[d] + DE_F * (pop[base_r1 + d] - pop[base_r2 + d]);
                break;
            case STRATEGY_RAND_TO_BEST_1:
                v = pop[base_r1 + d] + 0.5 * (best[d] - pop[base_r1 + d]) + DE_F * (pop[base_r2 + d] - pop[base_r3 + d]);
                break;
            default:
                v = target[d];
        }

        trial[d] = (rand_uniform(0.0, 1.0) < DE_CR || d == j_rand) ? v : target[d];
    }
}

void DE_optimize(Optimizer* opt, ObjectiveFunction objective_function, int strategy) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;

    // Flattened memory allocation
    double* pop = aligned_alloc(CACHE_LINE, sizeof(double) * pop_size * dim);
    double* fitness = aligned_alloc(CACHE_LINE, sizeof(double) * pop_size);
    double* best = aligned_alloc(CACHE_LINE, sizeof(double) * dim);
    double* trial = aligned_alloc(CACHE_LINE, sizeof(double) * dim);

    for (int i = 0; i < pop_size; i++) {
        int base = i * dim;
        for (int d = 0; d < dim; d++) {
            pop[base + d] = rand_uniform(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        fitness[i] = objective_function(&pop[base]);
    }

    // Initialize best
    memcpy(best, &pop[0], sizeof(double) * dim);
    double best_fit = fitness[0];
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fit) {
            best_fit = fitness[i];
            memcpy(best, &pop[i * dim], sizeof(double) * dim);
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            int base = i * dim;
            mutate_and_crossover(trial, &pop[base], pop, best, i, pop_size, dim, strategy);

            // Bound handling
            for (int d = 0; d < dim; d++) {
                double val = trial[d];
                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                if (unlikely(val < lower)) val = lower;
                if (unlikely(val > upper)) val = upper;
                trial[d] = val;
            }

            double trial_fit = objective_function(trial);
            if (trial_fit < fitness[i]) {
                memcpy(&pop[base], trial, sizeof(double) * dim);
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

    free(pop);
    free(fitness);
    free(best);
    free(trial);
}
