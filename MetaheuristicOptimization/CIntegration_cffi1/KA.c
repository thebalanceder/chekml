#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <omp.h> // Enable OpenMP

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

// Generate random double between [min, max]
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Fast in-place bounds check using restrict to help vectorize
static inline void enforce_bounds(double *restrict pos, const double *restrict lb, const double *restrict ub, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (pos[i] < lb[i]) pos[i] = lb[i];
        else if (pos[i] > ub[i]) pos[i] = ub[i];
    }
}

// Find nearest neighbor (min squared distance), return index
static inline int argmin_dist(Solution *restrict pop, int pop_size, const double *restrict target, int dim, int skip_idx) {
    double min_dist = DBL_MAX;
    int best_idx = -1;

    for (int i = 0; i < pop_size; ++i) {
        if (i == skip_idx) continue;
        double dist = 0.0;

        for (int d = 0; d < dim; ++d) {
            double diff = pop[i].position[d] - target[d];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

// Vectorized swirl operation
static inline void swirl(double *restrict pos, const double *restrict nn, const double *restrict lb, const double *restrict ub, int dim, int s, int s_max) {
    double strength = (s_max - s + 1.0) / s_max;
    for (int i = 0; i < dim; ++i) {
        pos[i] += strength * (nn[i] - pos[i]) * rand_double(-1.0, 1.0);
    }
    enforce_bounds(pos, lb, ub, dim);
}

// Weighted crossover
static inline void crossover_middle(const double *restrict a, const double *restrict b, const double *restrict c,
                                    double *restrict out, int dim, const double *restrict lb, const double *restrict ub) {
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

// ===============================
// Main Optimizer Function
// ===============================
void KA_optimize(Optimizer *opt, double (*objective)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    double *restrict lb = opt->bounds;
    double *restrict ub = &opt->bounds[dim];

    // Segment ratios
    double p1 = 0.2, p2 = 0.5;
    int s_max = 4;

    int m1 = (int)(p1 * pop_size);
    int m2 = 2 * ((int)(p2 * pop_size) / 2);
    int m3 = pop_size - m1 - m2;

    double *trial = (double *)aligned_alloc(64, sizeof(double) * dim); // aligned for SIMD

    // =======================
    // Initial fitness
    // =======================
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < pop_size; ++i) {
        double fit = objective(opt->population[i].position);
        opt->population[i].fitness = fit;

        #pragma omp critical
        {
            if (fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = fit;
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
            }
        }
    }

    // =======================
    // Main loop
    // =======================
    for (int iter = 0; iter < max_iter; ++iter) {

        // Group 1: Swirl-based exploitation
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m1; ++i) {
            int s = 1;
            int nn_idx = argmin_dist(opt->population, pop_size, opt->population[i].position, dim, i);

            while (s <= 2 * s_max - 1) {
                memcpy(trial, opt->population[i].position, sizeof(double) * dim);
                swirl(trial, opt->population[nn_idx].position, lb, ub, dim, s, s_max);

                double fit = objective(trial);
                if (fit < opt->population[i].fitness) {
                    memcpy(opt->population[i].position, trial, sizeof(double) * dim);
                    opt->population[i].fitness = fit;

                    #pragma omp critical
                    {
                        if (fit < opt->best_solution.fitness) {
                            opt->best_solution.fitness = fit;
                            memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                        }
                    }

                    nn_idx = argmin_dist(opt->population, pop_size, opt->population[i].position, dim, i);
                    s = 1;
                } else {
                    ++s;
                }
            }
        }

        // Group 2: Crossover
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m2; ++i) {
            int idx = m1 + i;
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

                #pragma omp critical
                {
                    if (fit < opt->best_solution.fitness) {
                        opt->best_solution.fitness = fit;
                        memcpy(opt->best_solution.position, trial, sizeof(double) * dim);
                    }
                }
            }
        }

        // Group 3: Random reinitialization
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m3; ++i) {
            int idx = m1 + m2 + i;
            for (int d = 0; d < dim; ++d)
                opt->population[idx].position[d] = rand_double(lb[d], ub[d]);

            double fit = objective(opt->population[idx].position);
            opt->population[idx].fitness = fit;

            #pragma omp critical
            {
                if (fit < opt->best_solution.fitness) {
                    opt->best_solution.fitness = fit;
                    memcpy(opt->best_solution.position, opt->population[idx].position, sizeof(double) * dim);
                }
            }
        }

        // Final bounds enforcement (safe guard)
        enforce_bound_constraints(opt);
    }

    free(trial);
}

