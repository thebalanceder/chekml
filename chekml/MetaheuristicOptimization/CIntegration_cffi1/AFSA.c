#include "AFSA.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

// === Inline clamp ===
static inline double clamp(double val, double min, double max) {
    return fmax(min, fmin(max, val));
}

// === Fast move_towards ===
void move_towards(const double *restrict Xi, const double *restrict Xj, int dim, double *restrict new_pos, const double *restrict lower_bounds, const double *restrict upper_bounds) {
    double norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = Xj[i] - Xi[i];
        norm_sq += diff * diff;
    }

    if (likely(norm_sq > 0.0)) {
        double inv_norm = STEP_SIZE / sqrt(norm_sq);
        double rand_scale = rand_double(0.0, 1.0) * inv_norm;
        for (int i = 0; i < dim; i++) {
            double step = (Xj[i] - Xi[i]) * rand_scale;
            new_pos[i] = clamp(Xi[i] + step, lower_bounds[i], upper_bounds[i]);
        }
    } else {
        memcpy(new_pos, Xi, sizeof(double) * dim);
    }
}

// === Prey behavior ===
void AFSA_prey_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *Xi = opt->population[index].position;
    double best_candidate[dim];
    memcpy(best_candidate, Xi, sizeof(double) * dim);
    double best_value = objective_function(Xi);

    for (int t = 0; t < TRY_NUMBER; t++) {
        double candidate[dim];
        for (int i = 0; i < dim; i++) {
            candidate[i] = clamp(
                Xi[i] + VISUAL * rand_double(-1.0, 1.0),
                opt->bounds[2 * i], opt->bounds[2 * i + 1]
            );
        }

        double val = objective_function(candidate);
        if (val < best_value) {
            memcpy(best_candidate, candidate, sizeof(double) * dim);
            best_value = val;
        }
    }

    if (best_value < objective_function(Xi)) {
        double new_pos[dim];
        move_towards(Xi, best_candidate, dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(Xi, new_pos, sizeof(double) * dim);
    }
}

// === Swarm behavior (avoid recomputing sqrt unless needed) ===
void AFSA_swarm_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *Xi = opt->population[index].position;
    double center[dim];
    memset(center, 0, sizeof(center));
    int count = 0;
    double visual_sq = VISUAL * VISUAL;

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist_sq = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist_sq += diff * diff;
        }
        if (dist_sq < visual_sq) {
            for (int d = 0; d < dim; d++) {
                center[d] += opt->population[j].position[d];
            }
            count++;
        }
    }

    if (unlikely(count == 0)) {
        AFSA_prey_behavior(opt, index, objective_function);
        return;
    }

    for (int d = 0; d < dim; d++) {
        center[d] /= count;
    }

    if ((objective_function(center) / count) < (objective_function(Xi) * DELTA)) {
        double new_pos[dim];
        move_towards(Xi, center, dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(Xi, new_pos, sizeof(double) * dim);
    } else {
        AFSA_prey_behavior(opt, index, objective_function);
    }
}

// === Follow behavior with faster distance ===
void AFSA_follow_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *Xi = opt->population[index].position;
    double best_neighbor[dim];
    memcpy(best_neighbor, Xi, sizeof(double) * dim);
    double best_value = objective_function(Xi);
    double visual_sq = VISUAL * VISUAL;

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist_sq = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist_sq += diff * diff;
        }

        if (dist_sq < visual_sq) {
            double val = objective_function(opt->population[j].position);
            if (val < best_value) {
                memcpy(best_neighbor, opt->population[j].position, sizeof(double) * dim);
                best_value = val;
            }
        }
    }

    if (best_value < (objective_function(Xi) * DELTA)) {
        double new_pos[dim];
        move_towards(Xi, best_neighbor, dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(Xi, new_pos, sizeof(double) * dim);
    } else {
        AFSA_prey_behavior(opt, index, objective_function);
    }
}

// === Main Optimizer Loop ===
void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int size = opt->population_size;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }

    enforce_bound_constraints(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int i = 0; i < size; i++) {
            int r = rand() % 3;
            if (r == 0) {
                AFSA_prey_behavior(opt, i, objective_function);
            } else if (r == 1) {
                AFSA_swarm_behavior(opt, i, objective_function);
            } else {
                AFSA_follow_behavior(opt, i, objective_function);
            }
        }

        for (int i = 0; i < size; i++) {
            double fit = objective_function(opt->population[i].position);
            if (fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = fit;
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
            }
        }

        enforce_bound_constraints(opt);
    }
}

