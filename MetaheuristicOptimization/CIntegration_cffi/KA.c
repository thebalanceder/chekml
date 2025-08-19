#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    int index;
    double fitness;
} IndexedSolution;


static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static void enforce_bounds(double *pos, double *lb, double *ub, int dim) {
    for (int i = 0; i < dim; i++) {
        if (pos[i] < lb[i]) pos[i] = lb[i];
        else if (pos[i] > ub[i]) pos[i] = ub[i];
    }
}

static int argmin_dist(Solution *pop, int pop_size, double *target, int dim, int skip_idx) {
    double min_dist = DBL_MAX;
    int nearest_idx = -1;

    for (int i = 0; i < pop_size; ++i) {
        if (i == skip_idx) continue;
        double dist = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = pop[i].position[d] - target[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

static void swirl(double *pos, double *nn, double *lb, double *ub, int dim, int s, int s_max) {
    double swirl_strength = (s_max - s + 1.0) / s_max;
    for (int i = 0; i < dim; ++i) {
        pos[i] += swirl_strength * (nn[i] - pos[i]) * rand_double(-1.0, 1.0);
    }
    enforce_bounds(pos, lb, ub, dim);
}

static void crossover_middle(double *a, double *b, double *c, double *out, int dim, double *lb, double *ub) {
    double w1 = rand_double(0, 1);
    double w2 = rand_double(0, 1);
    double w3 = rand_double(0, 1);
    double sum = w1 + w2 + w3;
    w1 /= sum; w2 /= sum; w3 /= sum;
    for (int i = 0; i < dim; ++i) {
        out[i] = w1 * a[i] + w2 * b[i] + w3 * c[i];
    }
    enforce_bounds(out, lb, ub, dim);
}

void KA_optimize(Optimizer *opt, double (*objective)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    double *lb = opt->bounds;
    double *ub = &opt->bounds[dim];

    int s_max = 4;
    double p1 = 0.2, p2 = 0.5;

    int m1 = (int)(p1 * pop_size);
    int m2 = 2 * ((int)(p2 * pop_size) / 2);
    int m3 = pop_size - m1 - m2;

    double trial[dim];

    // Initial fitness evaluation
    for (int i = 0; i < pop_size; ++i) {
        double fit = objective(opt->population[i].position);
        opt->population[i].fitness = fit;
        if (fit < opt->best_solution.fitness) {
            opt->best_solution.fitness = fit;
            memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
        }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        // m1: Best group with swirl
        for (int i = 0; i < m1; ++i) {
            int nn_idx = argmin_dist(opt->population, pop_size, opt->population[i].position, dim, i);
            int S = 1;
            while (S <= 2 * s_max - 1) {
                memcpy(trial, opt->population[i].position, sizeof(double) * dim);
                swirl(trial, opt->population[nn_idx].position, lb, ub, dim, S, s_max);
                double fit = objective(trial);
                if (fit < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, trial, sizeof(double) * dim);
                    opt->population[i].fitness = fit;
                    if (fit < opt->best_solution.fitness) {
                        opt->best_solution.fitness = fit;
                        memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                    }
                    nn_idx = argmin_dist(opt->population, pop_size, opt->population[i].position, dim, i);
                    S = 1;
                } else {
                    S++;
                }
            }
        }

        // m2: Middle group with crossover
        for (int i = 0; i < m2; ++i) {
            int idx = i + m1;
            int a = rand() % pop_size, b = rand() % pop_size;
            while (a == idx) a = rand() % pop_size;
            while (b == idx || b == a) b = rand() % pop_size;
            crossover_middle(opt->population[idx].position,
                             opt->population[a].position,
                             opt->population[b].position,
                             trial, dim, lb, ub);
            double fit = objective(trial);
            if (fit < opt->population[idx].fitness) {
                memcpy(opt->population[idx].position, trial, sizeof(double) * dim);
                opt->population[idx].fitness = fit;
                if (fit < opt->best_solution.fitness) {
                    opt->best_solution.fitness = fit;
                    memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                }
            }
        }

        // m3: Reinitialize worst group
        for (int i = 0; i < m3; ++i) {
            int idx = i + m1 + m2;
            for (int d = 0; d < dim; ++d)
                opt->population[idx].position[d] = rand_double(lb[d], ub[d]);
            double fit = objective(opt->population[idx].position);
            opt->population[idx].fitness = fit;
            if (fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = fit;
                memcpy(opt->best_solution.position, opt->population[idx].position, sizeof(double) * dim);
            }
        }

        // Enforce boundary
        enforce_bound_constraints(opt);
    }
}

