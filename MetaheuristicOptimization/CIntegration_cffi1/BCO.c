/* BCO.c - High-Performance CPU-Optimized Implementation of Bacterial Colony Optimization */
#include "BCO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

static inline double rand_gaussian() {
    double u1 = rand_double(1e-10, 1.0);
    double u2 = rand_double(0.0, 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static inline double compute_chemotaxis_step(int iteration, int max_iter) {
    return CHEMOTAXIS_STEP_MIN + (CHEMOTAXIS_STEP_MAX - CHEMOTAXIS_STEP_MIN) * ((double)(max_iter - iteration) / max_iter);
}

void chemotaxis_and_communication(Optimizer *opt, int iteration) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const double chemotaxis_step = compute_chemotaxis_step(iteration, opt->max_iter);

    for (int i = 0; i < pop_size; ++i) {
        double r = rand_double(0.0, 1.0);

        for (int j = 0; j < dim; ++j) {
            double diff = opt->best_solution.position[j] - opt->population[i].position[j];
            double step = chemotaxis_step * 0.5 * diff;
            if (r < 0.5) step += chemotaxis_step * rand_gaussian();
            opt->population[i].position[j] += step;
        }

        if (rand_double(0.0, 1.0) < COMMUNICATION_PROB) {
            int neighbor_idx = (rand_double(0.0, 1.0) < 0.5) ? (i + (rand() % 2 ? 1 : -1) + pop_size) % pop_size : rand() % pop_size;
            if (opt->population[neighbor_idx].fitness < opt->population[i].fitness) {
                memcpy(opt->population[i].position, opt->population[neighbor_idx].position, sizeof(double) * dim);
            } else if (opt->population[i].fitness > opt->best_solution.fitness) {
                memcpy(opt->population[i].position, opt->best_solution.position, sizeof(double) * dim);
            }
        }
    }
    enforce_bound_constraints(opt);
}

void elimination_and_reproduction(Optimizer *opt) {
    const int n = opt->population_size;
    const int dim = opt->dim;
    int num_elim = (int)(ELIMINATION_RATIO_BCO * n);
    int num_reproduce = num_elim / 2;

    for (int i = 0; i < num_elim; ++i) {
        int idx = n - 1 - i;
        double energy = 1.0 / (1.0 + opt->population[idx].fitness);
        if (energy < REPRODUCTION_THRESHOLD) {
            for (int j = 0; j < dim; ++j) {
                opt->population[idx].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
            opt->population[idx].fitness = INFINITY;
        }
    }

    for (int i = 0; i < num_reproduce; ++i) {
        int src = i;
        int dst = n - 1 - i;
        double energy = 1.0 / (1.0 + opt->population[src].fitness);
        if (energy >= REPRODUCTION_THRESHOLD) {
            memcpy(opt->population[dst].position, opt->population[src].position, sizeof(double) * dim);
        }
    }

    enforce_bound_constraints(opt);
}

void migration_phase(Optimizer *opt) {
    const int n = opt->population_size;
    const int dim = opt->dim;

    for (int i = 0; i < n; ++i) {
        if (rand_double(0.0, 1.0) < MIGRATION_PROBABILITY) {
            double norm = 0.0;
            for (int j = 0; j < dim; ++j) {
                double diff = opt->population[i].position[j] - opt->best_solution.position[j];
                norm += diff * diff;
            }
            norm = sqrt(norm);

            double energy = 1.0 / (1.0 + opt->population[i].fitness);
            if (energy < REPRODUCTION_THRESHOLD || norm < 1e-3) {
                for (int j = 0; j < dim; ++j) {
                    opt->population[i].position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
                }
                opt->population[i].fitness = INFINITY;
            }
        }
    }
    enforce_bound_constraints(opt);
}

void BCO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    const int dim = opt->dim;
    for (int j = 0; j < dim; ++j) {
        opt->best_solution.position[j] = opt->population[0].position[j];
    }
    opt->best_solution.fitness = objective_function(opt->best_solution.position);

    for (int iter = 0; iter < opt->max_iter; ++iter) {
        for (int i = 0; i < opt->population_size; ++i) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
            }
        }
        chemotaxis_and_communication(opt, iter);
        elimination_and_reproduction(opt);
        migration_phase(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}

