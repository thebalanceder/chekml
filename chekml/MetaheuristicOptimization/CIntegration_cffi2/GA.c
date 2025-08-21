#include "GA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// Random helpers
static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Clamp helper
static inline double clamp(double x, double min, double max) {
    return fmax(min, fmin(max, x));
}

void GA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    const int pop_size = opt->population_size;

    Solution* population = (Solution*)aligned_alloc(32, sizeof(Solution) * pop_size);
    for (int i = 0; i < pop_size; ++i) {
        population[i].position = (double*)aligned_alloc(32, sizeof(double) * dim);
        population[i].fitness = INFINITY;
    }

    Solution best_solution;
    best_solution.position = (double*)aligned_alloc(32, sizeof(double) * dim);
    best_solution.fitness = INFINITY;

    for (int p = 0; p < GA_NUM_POPULATIONS; ++p) {
        // === Initial population ===
        #pragma omp parallel for
        for (int i = 0; i < pop_size; ++i) {
            double* pos = population[i].position;
            for (int j = 0; j < dim; ++j) {
                pos[j] = rand_uniform(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            int best_idx = 0, worst_idx = 0;
            double best_fit = INFINITY;
            double worst_fit = -INFINITY;

            // === Parallel fitness evaluation + best/worst ===
            #pragma omp parallel for reduction(min:best_fit) reduction(max:worst_fit)
            for (int i = 0; i < pop_size; ++i) {
                double fit = objective_function(population[i].position);
                population[i].fitness = fit;

                if (fit < best_fit) {
                    #pragma omp critical
                    {
                        if (fit < best_fit) {
                            best_fit = fit;
                            best_idx = i;
                        }
                    }
                }
                if (fit > worst_fit) {
                    #pragma omp critical
                    {
                        if (fit > worst_fit) {
                            worst_fit = fit;
                            worst_idx = i;
                        }
                    }
                }
            }

            double* best = population[best_idx].position;
            double* worst = population[worst_idx].position;

            // === Crossover (uniform) and mutation (worst re-init) ===
            #pragma omp parallel for
            for (int i = 0; i < pop_size; ++i) {
                double* pos = population[i].position;
                if (i == best_idx || i == worst_idx) continue;

                for (int j = 0; j < dim; ++j) {
                    if (rand() % 2) {
                        int parent = rand() % 2 ? best_idx : (best_idx + 1) % pop_size;
                        pos[j] = population[parent].position[j];
                    }
                }
            }

            // === Mutation (reinitialize worst) ===
            #pragma omp parallel for
            for (int j = 0; j < dim; ++j) {
                worst[j] = rand_uniform(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }

            // === Track best overall ===
            if (best_fit < best_solution.fitness) {
                best_solution.fitness = best_fit;
                memcpy(best_solution.position, best, sizeof(double) * dim);
            }

            printf("GA|%5d -----> %9.16f\n", iter + 1, best_fit);
        }

        printf("GA Best in Pop %d: Fitness = %e\n", p + 1, best_solution.fitness);
    }

    memcpy(opt->best_solution.position, best_solution.position, sizeof(double) * dim);
    opt->best_solution.fitness = best_solution.fitness;

    for (int i = 0; i < pop_size; ++i) {
        free(population[i].position);
    }
    free(population);
    free(best_solution.position);
}

