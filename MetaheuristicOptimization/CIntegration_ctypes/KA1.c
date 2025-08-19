#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    int index;
    double fitness;
} IndexedSolution;


// Utility: random number in [min, max]
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Sort helper for population (used in qsort)
static int compare_by_fitness(const void *a, const void *b) {
    const IndexedSolution *ia = (const IndexedSolution *)a;
    const IndexedSolution *ib = (const IndexedSolution *)b;
    return (ia->fitness > ib->fitness) - (ia->fitness < ib->fitness);
}

// Find nearest neighbor (excluding self)
static int nearest_neighbor(Solution *population, int pop_size, int self_idx, int dim) {
    double min_dist = DBL_MAX;
    int nn_idx = -1;
    for (int i = 0; i < pop_size; i++) {
        if (i == self_idx) continue;
        double dist = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = population[i].position[d] - population[self_idx].position[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            nn_idx = i;
        }
    }
    return nn_idx;
}

// Swirl operator (targeted)
static void swirl(double *out, double *current, double *neighbor, int dim, int S, int S_max, double *lb, double *ub) {
    double swirl_strength = (S_max - S + 1.0) / S_max;
    for (int d = 0; d < dim; d++) {
        out[d] = current[d] + swirl_strength * (neighbor[d] - current[d]) * rand_double(-1.0, 1.0);
        if (out[d] < lb[d]) out[d] = lb[d];
        if (out[d] > ub[d]) out[d] = ub[d];
    }
}

// Crossover for middle group (blend of 3)
static void crossover_middle(double *p1, double *p2, double *p3, double *out, int dim, double *lb, double *ub) {
    double w1 = rand_double(0.0, 1.0);
    double w2 = rand_double(0.0, 1.0);
    double w3 = rand_double(0.0, 1.0);
    double sum = w1 + w2 + w3;
    w1 /= sum; w2 /= sum; w3 /= sum;

    for (int i = 0; i < dim; i++) {
        out[i] = w1 * p1[i] + w2 * p2[i] + w3 * p3[i];
        if (out[i] < lb[i]) out[i] = lb[i];
        if (out[i] > ub[i]) out[i] = ub[i];
    }
}

// Random solution initializer
static void random_solution(double *out, int dim, double *lb, double *ub) {
    for (int i = 0; i < dim; i++) {
        out[i] = rand_double(lb[i], ub[i]);
    }
}

void KA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    int S_max = 4;
    double *lb = opt->bounds;
    double *ub = &opt->bounds[dim];

    // Initialize population fitness
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
        }
    }

    // Indexed array to sort population
    IndexedSolution indexed[pop_size];

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            indexed[i].index = i;
            indexed[i].fitness = opt->population[i].fitness;
        }
        qsort(indexed, pop_size, sizeof(IndexedSolution), compare_by_fitness);

        int m1 = (int)(0.2 * pop_size);
        int m2 = (int)(0.5 * pop_size);
        m2 = m2 % 2 == 0 ? m2 : m2 - 1;  // ensure even
        int m3 = pop_size - (m1 + m2);

        // N1: Best group - targeted swirl
        for (int i = 0; i < m1; i++) {
            int idx = indexed[i].index;
            int nn_idx = nearest_neighbor(opt->population, pop_size, idx, dim);
            int S = 1;
            while (S <= 2 * S_max - 1) {
                double trial[dim];
                swirl(trial, opt->population[idx].position, opt->population[nn_idx].position, dim, S, S_max, lb, ub);
                double trial_fit = objective_function(trial);
                if (trial_fit < opt->population[idx].fitness) {
                    memcpy(opt->population[idx].position, trial, sizeof(double) * dim);
                    opt->population[idx].fitness = trial_fit;
                    if (trial_fit < opt->best_solution.fitness) {
                        opt->best_solution.fitness = trial_fit;
                        memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                    }
                    S = 1;  // restart
                } else {
                    S++;
                }
            }
        }

        // N2: Middle group - crossover
        for (int i = 0; i < m2; i++) {
            int idx = indexed[m1 + i].index;
            int a = rand() % pop_size, b = rand() % pop_size;
            double trial[dim];
            crossover_middle(opt->population[idx].position,
                             opt->population[a].position,
                             opt->population[b].position,
                             trial, dim, lb, ub);
            double fit = objective_function(trial);
            if (fit < opt->population[idx].fitness) {
                memcpy(opt->population[idx].position, trial, sizeof(double) * dim);
                opt->population[idx].fitness = fit;
                if (fit < opt->best_solution.fitness) {
                    opt->best_solution.fitness = fit;
                    memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                }
            }
        }

        // N3: Worst group - random reinit
        for (int i = 0; i < m3; i++) {
            int idx = indexed[m1 + m2 + i].index;
            double trial[dim];
            random_solution(trial, dim, lb, ub);
            double fit = objective_function(trial);
            if (fit < opt->population[idx].fitness) {
                memcpy(opt->population[idx].position, trial, sizeof(double) * dim);
                opt->population[idx].fitness = fit;
                if (fit < opt->best_solution.fitness) {
                    opt->best_solution.fitness = fit;
                    memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                }
            }
        }

        enforce_bound_constraints(opt);
        // Optionally: printf("Iter %d: Best = %.10f\n", iter + 1, opt->best_solution.fitness);
    }
}
