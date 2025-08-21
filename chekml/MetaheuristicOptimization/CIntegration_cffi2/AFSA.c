#include "AFSA.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

// === Helper: Clamp a value within bounds ===
double clamp(double val, double min, double max) {
    return (val < min) ? min : (val > max) ? max : val;
}

// === Move Xi towards Xj ===
void move_towards(const double *Xi, const double *Xj, int dim, double *new_pos, const double *lower_bounds, const double *upper_bounds) {
    double norm = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = Xj[i] - Xi[i];
        norm += diff * diff;
    }
    norm = sqrt(norm);
    if (norm == 0.0) {
        memcpy(new_pos, Xi, dim * sizeof(double));
        return;
    }
    for (int i = 0; i < dim; i++) {
        double step = STEP_SIZE_AFSA * (Xj[i] - Xi[i]) / norm * rand_double(0.0, 1.0);
        new_pos[i] = clamp(Xi[i] + step, lower_bounds[i], upper_bounds[i]);
    }
}

// === Prey Behavior ===
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

// === Swarm Behavior ===
void AFSA_swarm_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *Xi = opt->population[index].position;
    double center[dim];
    memset(center, 0, sizeof(center));
    int count = 0;

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < VISUAL) {
            for (int d = 0; d < dim; d++) {
                center[d] += opt->population[j].position[d];
            }
            count++;
        }
    }

    if (count == 0) {
        AFSA_prey_behavior(opt, index, objective_function);
        return;
    }

    for (int d = 0; d < dim; d++) {
        center[d] /= count;
    }

    if (objective_function(center) / count < objective_function(Xi) * DELTA) {
        double new_pos[dim];
        move_towards(Xi, center, dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(Xi, new_pos, sizeof(double) * dim);
    } else {
        AFSA_prey_behavior(opt, index, objective_function);
    }
}

// === Follow Behavior ===
void AFSA_follow_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *Xi = opt->population[index].position;
    double best_neighbor[dim];
    memcpy(best_neighbor, Xi, sizeof(double) * dim);
    double best_value = objective_function(Xi);

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < VISUAL) {
            double val = objective_function(opt->population[j].position);
            if (val < best_value) {
                memcpy(best_neighbor, opt->population[j].position, sizeof(double) * dim);
                best_value = val;
            }
        }
    }

    if (best_value < objective_function(Xi) * DELTA) {
        double new_pos[dim];
        move_towards(Xi, best_neighbor, dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(Xi, new_pos, sizeof(double) * dim);
    } else {
        AFSA_prey_behavior(opt, index, objective_function);
    }
}

// === Main Optimize ===
void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }

    enforce_bound_constraints(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int i = 0; i < opt->population_size; i++) {
            switch (rand() % 3) {
                case 0:
                    AFSA_prey_behavior(opt, i, objective_function);
                    break;
                case 1:
                    AFSA_swarm_behavior(opt, i, objective_function);
                    break;
                case 2:
                    AFSA_follow_behavior(opt, i, objective_function);
                    break;
            }
        }

        for (int i = 0; i < opt->population_size; i++) {
            double fit = objective_function(opt->population[i].position);
            if (fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = fit;
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * opt->dim);
            }
        }

        enforce_bound_constraints(opt);
    }
}
