// GA.c
#include "GA.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline double clamp(double x, double min, double max) {
    return fmax(min, fmin(max, x));
}

void GA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    const int dim = opt->dim;
    const int num_chromosomes = opt->population_size;
    const int max_iter = opt->max_iter;

    double* pop = (double*)malloc(num_chromosomes * dim * sizeof(double));
    double* fitness = (double*)malloc(num_chromosomes * sizeof(double));
    double* best_sol = (double*)malloc(dim * sizeof(double));

    double best_fit = INFINITY;

    // Outer population loop (simulates MATLAB's NumPop)
    for (int outer = 0; outer < GA_NUM_POPULATIONS; outer++) {
        // === Initialization ===
        for (int i = 0; i < num_chromosomes; i++) {
            for (int d = 0; d < dim; d++) {
                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                pop[i * dim + d] = rand_uniform(lower, upper);
            }
        }

        // === Iterative Evolution ===
        for (int iter = 0; iter < max_iter; iter++) {
            // Evaluate fitness
            #pragma omp parallel for
            for (int i = 0; i < num_chromosomes; i++) {
                fitness[i] = objective_function(&pop[i * dim]);
            }

            // Natural selection (elitism)
            int best_idx = 0;
            double min_fit = fitness[0];
            for (int i = 1; i < num_chromosomes; i++) {
                if (fitness[i] < min_fit) {
                    min_fit = fitness[i];
                    best_idx = i;
                }
            }
            memcpy(best_sol, &pop[best_idx * dim], dim * sizeof(double));

            // Update global best
            if (min_fit < best_fit) {
                best_fit = min_fit;
                memcpy(opt->best_solution.position, best_sol, dim * sizeof(double));
                opt->best_solution.fitness = best_fit;
            }

#if GA_VERBOSITY
            printf("GA Iteration %d: Best Fitness = %f\n", iter + 1, min_fit);
#endif

            // Uniform Crossover
            for (int i = 0; i < num_chromosomes; i++) {
                if (i == best_idx) continue;
                for (int d = 0; d < dim; d++) {
                    int coin = rand() % 2;
                    if (coin) {
                        int parent = (rand() % 2 == 0) ? best_idx : (best_idx + 1) % num_chromosomes;
                        pop[i * dim + d] = pop[parent * dim + d];
                    }
                }
            }

            // Mutation (on worst individual)
            int worst_idx = 0;
            double worst_fit = fitness[0];
            for (int i = 1; i < num_chromosomes; i++) {
                if (fitness[i] > worst_fit) {
                    worst_fit = fitness[i];
                    worst_idx = i;
                }
            }
            for (int d = 0; d < dim; d++) {
                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                pop[worst_idx * dim + d] = rand_uniform(lower, upper);
            }
        }
    }

    free(pop);
    free(fitness);
    free(best_sol);
}

