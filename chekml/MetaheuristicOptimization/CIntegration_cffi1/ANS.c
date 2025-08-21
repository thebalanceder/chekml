#include "ANS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>  // For parallelism

// Inline clamp
static inline double clamp(double x, double min, double max) {
    return fmax(min, fmin(max, x));
}

// Uniform random number in [min, max]
static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    const int num_vars = opt->dim;
    const int num_neighborhoods = ANS_NUM_NEIGHBORHOODS;
    const int max_iterations = opt->max_iter;
    const double mutation_rate = ANS_MUTATION_RATE;

    // Allocate memory
    double* populations = (double*)malloc(num_neighborhoods * num_vars * sizeof(double));
    double* fitness = (double*)malloc(num_neighborhoods * sizeof(double));

    // === Initialization (Serial) ===
    for (int i = 0; i < num_neighborhoods; i++) {
        double* p = populations + i * num_vars;
        for (int d = 0; d < num_vars; d++) {
            double lower = opt->bounds[2 * d];
            double upper = opt->bounds[2 * d + 1];
            p[d] = rand_uniform(lower, upper);
        }
    }

    // Declare these outside loop so we can use them at the end
    int best_idx = 0;
    double best_fit = INFINITY;

    // === Optimization Loop ===
    for (int iter = 0; iter < max_iterations; iter++) {
        best_fit = INFINITY;

        // === Evaluate fitness in parallel ===
        #pragma omp parallel for shared(populations, fitness)
        for (int i = 0; i < num_neighborhoods; i++) {
            double* p = populations + i * num_vars;
            fitness[i] = objective_function(p);
        }

        // === Find best solution (serial) ===
        for (int i = 0; i < num_neighborhoods; i++) {
            if (fitness[i] < best_fit) {
                best_fit = fitness[i];
                best_idx = i;
            }
        }

#if VERBOSE
        printf("ANS Iteration %d: Best Fitness = %f\n", iter + 1, best_fit);
#endif

        // === Move neighborhoods in parallel ===
        #pragma omp parallel for shared(populations)
        for (int i = 0; i < num_neighborhoods; i++) {
            int neighbor_index;
            do {
                neighbor_index = rand() % num_neighborhoods;
            } while (neighbor_index == i);

            double* p = populations + i * num_vars;
            double* neighbor = populations + neighbor_index * num_vars;

            for (int d = 0; d < num_vars; d++) {
                double direction = neighbor[d] - p[d];
                p[d] += mutation_rate * direction;

                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                p[d] = clamp(p[d], lower, upper);
            }
        }
    }

    // === Final best solution ===
    double* best = populations + best_idx * num_vars;
    memcpy(opt->best_solution.position, best, num_vars * sizeof(double));
    opt->best_solution.fitness = fitness[best_idx];

    free(populations);
    free(fitness);
}
