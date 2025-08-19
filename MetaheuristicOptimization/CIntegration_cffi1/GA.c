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

// Safe aligned allocation
static inline void* safe_aligned_alloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 32, size) != 0) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void GA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    const int pop_size = opt->population_size;

    // === Allocate population ===
    Solution* population = (Solution*)safe_aligned_alloc(sizeof(Solution) * pop_size);
    for (int i = 0; i < pop_size; ++i) {
        population[i].position = (double*)safe_aligned_alloc(sizeof(double) * dim);
        population[i].fitness = INFINITY;
    }

    Solution best_solution;
    best_solution.position = (double*)safe_aligned_alloc(sizeof(double) * dim);
    best_solution.fitness = INFINITY;

    for (int p = 0; p < GA_NUM_POPULATIONS; ++p) {
        // === Initialize population ===
        #pragma omp parallel for
        for (int i = 0; i < pop_size; ++i) {
            double* pos = population[i].position;
            for (int j = 0; j < dim; ++j) {
                pos[j] = rand_uniform(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            int best_idx = -1, worst_idx = -1;
            double best_fit = INFINITY, worst_fit = -INFINITY;

            // === Parallel fitness evaluation ===
            #pragma omp parallel
            {
                int local_best = -1, local_worst = -1;
                double local_best_fit = INFINITY, local_worst_fit = -INFINITY;

                #pragma omp for nowait
                for (int i = 0; i < pop_size; ++i) {
                    double fit = objective_function(population[i].position);
                    population[i].fitness = fit;

                    if (fit < local_best_fit) {
                        local_best_fit = fit;
                        local_best = i;
                    }
                    if (fit > local_worst_fit) {
                        local_worst_fit = fit;
                        local_worst = i;
                    }
                }

                #pragma omp critical
                {
                    if (local_best_fit < best_fit) {
                        best_fit = local_best_fit;
                        best_idx = local_best;
                    }
                    if (local_worst_fit > worst_fit) {
                        worst_fit = local_worst_fit;
                        worst_idx = local_worst;
                    }
                }
            }

            // === Crossover (uniform) ===
            double* best = population[best_idx].position;
            #pragma omp parallel for
            for (int i = 0; i < pop_size; ++i) {
                if (i == best_idx || i == worst_idx) continue;
                double* pos = population[i].position;
                for (int j = 0; j < dim; ++j) {
                    if (rand() % 2) {
                        pos[j] = best[j];
                    }
                }
            }

            // === Mutation (reinitialize worst) ===
            double* worst = population[worst_idx].position;
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

    // === Final result ===
    memcpy(opt->best_solution.position, best_solution.position, sizeof(double) * dim);
    opt->best_solution.fitness = best_solution.fitness;

    for (int i = 0; i < pop_size; ++i) {
        free(population[i].position);
    }
    free(population);
    free(best_solution.position);
}
