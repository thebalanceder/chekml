#ifndef BSA_H
#define BSA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>  // AVX2 intrinsics
#include "generaloptimizer.h"

#define BSA_DIM_RATE 0.5

// Function declarations
double rand_double(double min, double max);
void BSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif /* BSA_H */

/* BSA.c - Optimized Backtracking Search Algorithm with AVX2 */
#include "BSA.h"
#include <time.h>

static inline double get_scale_factor() {
    return 3.0 * rand_double(-1.0, 1.0);  // Brownian-walk factor
}

static inline void generate_population(Optimizer *opt, Solution *pop) {
    int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = pop[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

static inline void boundary_control(Optimizer *opt) {
    int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = opt->bounds[2 * j];
            double upper = opt->bounds[2 * j + 1];
            if (pos[j] < lower || pos[j] > upper) {
                pos[j] = rand_double(lower, upper);
            }
        }
    }
}

void BSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int max_iter = opt->max_iter;

    Solution *historical_pop = (Solution *)aligned_alloc(32, sizeof(Solution) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        historical_pop[i].position = (double *)aligned_alloc(32, sizeof(double) * dim);
        historical_pop[i].fitness = INFINITY;
    }

    generate_population(opt, opt->population);
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    generate_population(opt, historical_pop);

    for (int iter = 0; iter < max_iter; iter++) {
        if (rand_double(0.0, 1.0) < rand_double(0.0, 1.0)) {
            for (int i = 0; i < pop_size; i++) {
                __m256d *src = (__m256d *)opt->population[i].position;
                __m256d *dst = (__m256d *)historical_pop[i].position;
                for (int j = 0; j < dim / 4; j++) {
                    dst[j] = src[j];
                }
            }
        }

        for (int i = pop_size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            Solution temp = historical_pop[i];
            historical_pop[i] = historical_pop[j];
            historical_pop[j] = temp;
        }

        double F = get_scale_factor();

        for (int i = 0; i < pop_size; i++) {
            int count = (rand_double(0.0, 1.0) < rand_double(0.0, 1.0)) ?
                        (int)(ceil(BSA_DIM_RATE * rand_double(0.0, 1.0) * dim)) : 1;

            for (int c = 0; c < count; c++) {
                int j = rand() % dim;
                opt->population[i].position[j] += F * (historical_pop[i].position[j] - opt->population[i].position[j]);
            }
        }

        boundary_control(opt);

        for (int i = 0; i < pop_size; i++) {
            double new_fit = objective_function(opt->population[i].position);
            if (new_fit < opt->population[i].fitness) {
                opt->population[i].fitness = new_fit;
                if (new_fit < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fit;
                    __m256d *src = (__m256d *)opt->population[i].position;
                    __m256d *dst = (__m256d *)opt->best_solution.position;
                    for (int j = 0; j < dim / 4; j++) {
                        dst[j] = src[j];
                    }
                }
            }
        }

        printf("BSA|%5d -----> %9.16f\n", iter + 1, opt->best_solution.fitness);
        enforce_bound_constraints(opt);
    }

    for (int i = 0; i < pop_size; i++) free(historical_pop[i].position);
    free(historical_pop);
}
