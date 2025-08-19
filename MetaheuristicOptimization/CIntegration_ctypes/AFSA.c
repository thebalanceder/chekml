/* AFSA.c - Implementation file for Artificial Fish Swarm Algorithm */
#include "AFSA.h"
#include <string.h> // for memcpy

// Move fish Xi towards Xj
void move_towards(double *Xi, double *Xj, int dim, double *new_pos, double *lower_bounds, double *upper_bounds) {
    double direction[dim];
    double norm = 0.0;
    for (int i = 0; i < dim; i++) {
        direction[i] = Xj[i] - Xi[i];
        norm += direction[i] * direction[i];
    }
    norm = sqrt(norm);

    if (norm == 0.0) {
        memcpy(new_pos, Xi, dim * sizeof(double));
        return;
    }

    for (int i = 0; i < dim; i++) {
        double step_val = STEP_SIZE * direction[i] / norm * rand_double(0.0, 1.0);
        new_pos[i] = Xi[i] + step_val;
        if (new_pos[i] < lower_bounds[i]) new_pos[i] = lower_bounds[i];
        if (new_pos[i] > upper_bounds[i]) new_pos[i] = upper_bounds[i];
    }
}

// Prey behavior
void prey_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    double *Xi = opt->population[index].position;
    double best_candidate[opt->dim];
    memcpy(best_candidate, Xi, opt->dim * sizeof(double));
    double best_value = objective_function(Xi);

    for (int t = 0; t < TRY_NUMBER; t++) {
        double candidate[opt->dim];
        for (int i = 0; i < opt->dim; i++) {
            candidate[i] = Xi[i] + VISUAL * (rand_double(-1.0, 1.0));
            if (candidate[i] < opt->bounds[2 * i]) candidate[i] = opt->bounds[2 * i];
            if (candidate[i] > opt->bounds[2 * i + 1]) candidate[i] = opt->bounds[2 * i + 1];
        }

        double candidate_value = objective_function(candidate);
        if (candidate_value < best_value) {
            memcpy(best_candidate, candidate, opt->dim * sizeof(double));
            best_value = candidate_value;
        }
    }

    double new_pos[opt->dim];
    if (best_value < objective_function(Xi)) {
        move_towards(Xi, best_candidate, opt->dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(opt->population[index].position, new_pos, opt->dim * sizeof(double));
    }
}

// Swarm behavior
void swarm_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    double *Xi = opt->population[index].position;
    double center[opt->dim];
    int count = 0;
    memset(center, 0, sizeof(center));

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist = 0.0;
        for (int d = 0; d < opt->dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < VISUAL) {
            for (int d = 0; d < opt->dim; d++) {
                center[d] += opt->population[j].position[d];
            }
            count++;
        }
    }

    if (count == 0) {
        prey_behavior(opt, index, objective_function);
        return;
    }

    for (int d = 0; d < opt->dim; d++) {
        center[d] /= count;
    }
    double center_value = objective_function(center);

    if (center_value / count < objective_function(Xi) * DELTA) {
        double new_pos[opt->dim];
        move_towards(Xi, center, opt->dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(opt->population[index].position, new_pos, opt->dim * sizeof(double));
    } else {
        prey_behavior(opt, index, objective_function);
    }
}

// Follow behavior
void follow_behavior(Optimizer *opt, int index, double (*objective_function)(double *)) {
    double *Xi = opt->population[index].position;
    double best_neighbor[opt->dim];
    memcpy(best_neighbor, Xi, opt->dim * sizeof(double));
    double best_value = objective_function(Xi);

    for (int j = 0; j < opt->population_size; j++) {
        if (j == index) continue;
        double dist = 0.0;
        for (int d = 0; d < opt->dim; d++) {
            double diff = opt->population[j].position[d] - Xi[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < VISUAL) {
            double neighbor_value = objective_function(opt->population[j].position);
            if (neighbor_value < best_value) {
                memcpy(best_neighbor, opt->population[j].position, opt->dim * sizeof(double));
                best_value = neighbor_value;
            }
        }
    }

    if (best_value < objective_function(Xi) * DELTA) {
        double new_pos[opt->dim];
        move_towards(Xi, best_neighbor, opt->dim, new_pos, &opt->bounds[0], &opt->bounds[1]);
        memcpy(opt->population[index].position, new_pos, opt->dim * sizeof(double));
    } else {
        prey_behavior(opt, index, objective_function);
    }
}

// Main AFSA Optimization
void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
    enforce_bound_constraints(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int i = 0; i < opt->population_size; i++) {
            int r = rand() % 3;
            if (r == 0) prey_behavior(opt, i, objective_function);
            else if (r == 1) swarm_behavior(opt, i, objective_function);
            else follow_behavior(opt, i, objective_function);
        }

        for (int i = 0; i < opt->population_size; i++) {
            double fit = objective_function(opt->population[i].position);
            if (fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = fit;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }
        enforce_bound_constraints(opt);
    }
}
