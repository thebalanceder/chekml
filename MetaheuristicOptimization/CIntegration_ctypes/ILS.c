// ILS.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "generaloptimizer.h"
#include "ILS.h"

static int in_bounds(double* point, double* bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        if (point[i] < bounds[2 * i] || point[i] > bounds[2 * i + 1]) {
            return 0;
        }
    }
    return 1;
}

static void random_point_within_bounds(double* point, double* bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double min = bounds[2 * i];
        double max = bounds[2 * i + 1];
        point[i] = min + ((double)rand() / RAND_MAX) * (max - min);
    }
}

static void hill_climb(double* solution, double* bounds, int dim, int n_iterations, double step_size, ObjectiveFunction objective_function, double* best_eval) {
    double* candidate = (double*)malloc(dim * sizeof(double));
    double* current = (double*)malloc(dim * sizeof(double));
    memcpy(current, solution, dim * sizeof(double));
    *best_eval = objective_function(current);

    for (int i = 0; i < n_iterations; i++) {
        int valid = 0;
        while (!valid) {
            for (int j = 0; j < dim; j++) {
                candidate[j] = current[j] + ((double)rand() / RAND_MAX) * step_size * 2 - step_size;
            }
            valid = in_bounds(candidate, bounds, dim);
        }

        double candidate_eval = objective_function(candidate);
        if (candidate_eval <= *best_eval) {
            memcpy(current, candidate, dim * sizeof(double));
            *best_eval = candidate_eval;
        }
    }
    memcpy(solution, current, dim * sizeof(double));
    free(candidate);
    free(current);
}

void ILS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Invalid optimizer or objective function.\n");
        return;
    }

    int dim = opt->dim;
    int n_restarts = 30;
    int n_iterations = 1000;
    double step_size = 0.05;
    double perturbation_size = 1.0;

    double* best_solution = (double*)malloc(dim * sizeof(double));
    double* current = (double*)malloc(dim * sizeof(double));
    double* temp = (double*)malloc(dim * sizeof(double));

    if (!best_solution || !current || !temp) {
        fprintf(stderr, "Memory allocation failed\n");
        free(best_solution);
        free(current);
        free(temp);
        return;
    }

    random_point_within_bounds(best_solution, opt->bounds, dim);
    double best_fitness = objective_function(best_solution);

    for (int r = 0; r < n_restarts; r++) {
        int valid = 0;
        while (!valid) {
            for (int i = 0; i < dim; i++) {
                temp[i] = best_solution[i] + ((double)rand() / RAND_MAX) * perturbation_size * 2 - perturbation_size;
            }
            valid = in_bounds(temp, opt->bounds, dim);
        }

        double eval;
        hill_climb(temp, opt->bounds, dim, n_iterations, step_size, objective_function, &eval);

        if (eval < best_fitness) {
            best_fitness = eval;
            memcpy(best_solution, temp, dim * sizeof(double));
            printf("Restart %d: Best Fitness = %.5f\n", r, best_fitness);
        }
    }

    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, best_solution, dim * sizeof(double));

    free(best_solution);
    free(current);
    free(temp);
}

