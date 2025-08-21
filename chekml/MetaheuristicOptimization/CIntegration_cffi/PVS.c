#include "PVS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define M_PI 3.14159265358979323846

// Simplified gamma inverse approximation (faster but approximate)
double approx_gammaincinv(double x, double a) {
    return 1.0 / (x * (0.5 + 0.5 * a)); // Linear approximation
}

// Initialize PVS data and vortex
void initialize_vortex(Optimizer *opt, PVSData *data) {
    int i, j;
    double x = X_GAMMA;
    double a = 1.0;
    double ginv = (1.0 / x) * approx_gammaincinv(x, a);

    // Allocate PVS data
    data->center = (double *)malloc(opt->dim * sizeof(double));
    data->obj_vals = (double *)malloc(opt->population_size * sizeof(double));
    data->prob = (double *)malloc(opt->population_size * sizeof(double));
    data->sol = (double *)malloc(opt->dim * sizeof(double));
    data->mutated = (double *)malloc(opt->dim * sizeof(double));
    data->bound_diffs = (double *)malloc(opt->dim * sizeof(double));
    data->prob_mut = 1.0 / opt->dim;
    data->prob_cross = 1.0 / opt->dim;

    // Precompute bound differences
    for (j = 0; j < opt->dim; j++) {
        data->bound_diffs[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
        data->center[j] = 0.5 * (opt->bounds[2 * j] + opt->bounds[2 * j + 1]);
    }
    opt->best_solution.fitness = INFINITY;

    // Initialize candidates with normal distribution
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            double radius = ginv * data->bound_diffs[j] / 2.0;
            double u1 = rand_double(0.0, 1.0);
            double u2 = rand_double(0.0, 1.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            opt->population[i].position[j] = data->center[j] + radius * z;
        }
    }
    enforce_bound_constraints(opt);
}

// First Phase: Generate candidate solutions
void first_phase(Optimizer *opt, PVSData *data, int iteration, double radius) {
    int i, j, size = (iteration == 0) ? opt->population_size : opt->population_size / 2;
    double z1, z2, u1, u2;
    int use_z2 = 0;

    for (i = 0; i < size; i++) {
        for (j = 0; j < opt->dim; j++) {
            if (!use_z2) {
                u1 = rand_double(0.0, 1.0);
                u2 = rand_double(0.0, 1.0);
                double r = sqrt(-2.0 * log(u1));
                z1 = r * cos(2.0 * M_PI * u2);
                z2 = r * sin(2.0 * M_PI * u2);
                opt->population[i].position[j] = data->center[j] + radius * z1;
                use_z2 = 1;
            } else {
                opt->population[i].position[j] = data->center[j] + radius * z2;
                use_z2 = 0;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Polynomial Mutation
void polynomial_mutation(Optimizer *opt, PVSData *data, double *restrict solution, double *restrict mutated, int *restrict state) {
    int i;
    double mut_pow = 1.0 / (1.0 + DISTRIBUTION_INDEX);
    *state = 0;

    for (i = 0; i < opt->dim; i++) {
        mutated[i] = solution[i];
        if (rand_double(0.0, 1.0) <= data->prob_mut) {
            double y = solution[i];
            double yL = opt->bounds[2 * i];
            double yU = opt->bounds[2 * i + 1];
            double delta1 = (y - yL) / data->bound_diffs[i];
            double delta2 = (yU - y) / data->bound_diffs[i];
            double rnd = rand_double(0.0, 1.0);
            double xy, val, deltaq;

            if (rnd <= 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * pow(xy, DISTRIBUTION_INDEX + 1);
                deltaq = pow(val, mut_pow) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * pow(xy, DISTRIBUTION_INDEX + 1);
                deltaq = 1.0 - pow(val, mut_pow);
            }

            y = y + deltaq * data->bound_diffs[i];
            mutated[i] = (y < yL) ? yL : (y > yU) ? yU : y;
            (*state)++;
        }
    }
}

// Second Phase: Crossover and Mutation
void second_phase(Optimizer *opt, PVSData *data, int iteration, ObjectiveFunction objective_function) {
    int i, j, neighbor, param2change;
    double *restrict obj_vals = data->obj_vals;
    double *restrict prob = data->prob;
    double *restrict sol = data->sol;
    double *restrict mutated = data->mutated;
    int state;

    // Update fitness values
    for (i = 0; i < opt->population_size; i++) {
        obj_vals[i] = opt->population[i].fitness;
    }

    // Compute roulette wheel probabilities
    double max_val = obj_vals[0];
    for (i = 1; i < opt->population_size; i++) {
        if (obj_vals[i] > max_val) max_val = obj_vals[i];
    }
    double sum_prob = 0.0;
    for (i = 0; i < opt->population_size; i++) {
        prob[i] = 0.9 * (max_val - obj_vals[i]) + 0.1;
        sum_prob += prob[i];
    }
    prob[0] /= sum_prob;
    for (i = 1; i < opt->population_size; i++) {
        prob[i] = prob[i - 1] + (prob[i] / sum_prob);
    }

    // Process second half of population
    for (i = opt->population_size / 2; i < opt->population_size; i++) {
        // Roulette wheel selection
        double r = rand_double(0.0, 1.0);
        neighbor = 0;
        for (j = 0; j < opt->population_size - 1; j++) {
            if (r <= prob[j]) {
                neighbor = j;
                break;
            }
        }
        while (i == neighbor) {
            r = rand_double(0.0, 1.0);
            neighbor = 0;
            for (j = 0; j < opt->population_size - 1; j++) {
                if (r <= prob[j]) {
                    neighbor = j;
                    break;
                }
            }
        }

        // Crossover
        double *pop_i = opt->population[i].position;
        double *pop_n = opt->population[neighbor].position;
        param2change = rand() % opt->dim;
        for (j = 0; j < opt->dim; j++) {
            sol[j] = pop_i[j];
            if (rand_double(0.0, 1.0) < data->prob_cross || j == param2change) {
                sol[j] += (pop_i[j] - pop_n[j]) * (rand_double(0.0, 1.0) - 0.5) * 2.0;
                sol[j] = (sol[j] < opt->bounds[2 * j]) ? opt->bounds[2 * j] :
                         (sol[j] > opt->bounds[2 * j + 1]) ? opt->bounds[2 * j + 1] : sol[j];
            }
        }

        // Evaluate new solution
        double obj_val_sol = objective_function(sol);
        if (obj_val_sol < obj_vals[i]) {
            memcpy(pop_i, sol, opt->dim * sizeof(double));
            opt->population[i].fitness = obj_val_sol;
            obj_vals[i] = obj_val_sol;
        } else {
            polynomial_mutation(opt, data, pop_i, mutated, &state);
            if (state > 0) {
                double obj_val_mut = objective_function(mutated);
                if (obj_val_mut < obj_vals[i]) {
                    memcpy(pop_i, mutated, opt->dim * sizeof(double));
                    opt->population[i].fitness = obj_val_mut;
                    obj_vals[i] = obj_val_mut;
                }
            }
        }
    }
}

// Main Optimization Function
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int iteration = 0;
    double x = X_GAMMA;
    PVSData data = {0};

    initialize_vortex(opt, &data);

    while (iteration < opt->max_iter) {
        // Update radius
        double a = (opt->max_iter - iteration) / (double)opt->max_iter;
        a = fmax(a, 0.1);
        double ginv = (1.0 / x) * approx_gammaincinv(x, a);
        double radius = ginv * data.bound_diffs[0] / 2.0;

        // First phase
        first_phase(opt, &data, iteration, radius);
        int size = (iteration == 0) ? opt->population_size : opt->population_size / 2;
        for (int i = 0; i < size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                memcpy(data.center, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Second phase
        second_phase(opt, &data, iteration, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                memcpy(data.center, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        iteration++;
    }

    // Cleanup
    free(data.center);
    free(data.obj_vals);
    free(data.prob);
    free(data.sol);
    free(data.mutated);
    free(data.bound_diffs);
}
