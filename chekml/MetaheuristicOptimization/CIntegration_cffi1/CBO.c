/* CBO.c - Ultra-Optimized Implementation for Colliding Bodies Optimization */
/* Compile with: gcc -O3 -ffast-math -mavx2 -funroll-loops -fopenmp */
/* Disable OpenMP with -DNO_OPENMP if single-threaded is strictly required */
#include "CBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h> /* For memcpy */

/* Assume max dimension for stack allocation; adjust if needed */
#define MAX_DIM 128

/* Collision Phase (Movement towards center of mass) */
void collision_phase(Optimizer *opt) {
    double center_of_mass[MAX_DIM];
    double inv_pop_size = 1.0 / opt->population_size;

    /* Initialize center of mass */
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        center_of_mass[j] = 0.0;
    }

    /* Compute center of mass with SIMD-friendly loop */
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            center_of_mass[j] += pos[j];
        }
    }
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        center_of_mass[j] *= inv_pop_size;
    }

    /* Update positions and enforce bounds with loop unrolling */
    #ifndef NO_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double min_bounds[MAX_DIM], max_bounds[MAX_DIM];
        /* Preload bounds for cache efficiency */
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            min_bounds[j] = opt->bounds[2 * j];
            max_bounds[j] = opt->bounds[2 * j + 1];
        }

        /* Unroll inner loop for small dimensions */
        int j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            pos[j] += ALPHA * (center_of_mass[j] - pos[j]);
            pos[j] = pos[j] < min_bounds[j] ? min_bounds[j] : pos[j] > max_bounds[j] ? max_bounds[j] : pos[j];
            pos[j + 1] += ALPHA * (center_of_mass[j + 1] - pos[j + 1]);
            pos[j + 1] = pos[j + 1] < min_bounds[j + 1] ? min_bounds[j + 1] : pos[j + 1] > max_bounds[j + 1] ? max_bounds[j + 1] : pos[j + 1];
            pos[j + 2] += ALPHA * (center_of_mass[j + 2] - pos[j + 2]);
            pos[j + 2] = pos[j + 2] < min_bounds[j + 2] ? min_bounds[j + 2] : pos[j + 2] > max_bounds[j + 2] ? max_bounds[j + 2] : pos[j + 2];
            pos[j + 3] += ALPHA * (center_of_mass[j + 3] - pos[j + 3]);
            pos[j + 3] = pos[j + 3] < min_bounds[j + 3] ? min_bounds[j + 3] : pos[j + 3] > max_bounds[j + 3] ? max_bounds[j + 3] : pos[j + 3];
        }
        for (; j < opt->dim; j++) {
            pos[j] += ALPHA * (center_of_mass[j] - pos[j]);
            pos[j] = pos[j] < min_bounds[j] ? min_bounds[j] : pos[j] > max_bounds[j] ? max_bounds[j] : pos[j];
        }
    }
}

/* Update Best Solution */
void update_best_solution(Optimizer *opt, double (*objective_function)(double *)) {
    double best_fitness = opt->best_solution.fitness;
    double best_pos[MAX_DIM];

    /* Cache best position */
    memcpy(best_pos, opt->best_solution.position, opt->dim * sizeof(double));

    #ifndef NO_OPENMP
    #pragma omp parallel for schedule(static) reduction(min:best_fitness)
    #endif
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        if (new_fitness < best_fitness) {
            #ifndef NO_OPENMP
            #pragma omp critical
            #endif
            {
                if (new_fitness < best_fitness) {
                    best_fitness = new_fitness;
                    memcpy(best_pos, opt->population[i].position, opt->dim * sizeof(double));
                }
            }
        }
    }

    /* Update best solution */
    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, best_pos, opt->dim * sizeof(double));
}

/* Main Optimization Function */
void CBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL));
    for (int iter = 0; iter < opt->max_iter; iter++) {
        collision_phase(opt);
        update_best_solution(opt, objective_function);
    }
}
