#include "PVS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Approximate inverse gamma function for radius calculation (simplified)
double approx_gammaincinv(double x, double a) {
    // Simplified approximation: use a scaled random gamma variate
    return 1.0 / (x * rand_double(0.1, 1.0));
}

// Initialize the vortex center and candidate solutions
void initialize_vortex(Optimizer *opt) {
    int i, j;
    double x = X_GAMMA;
    double a = 1.0;
    double ginv = (1.0 / x) * approx_gammaincinv(x, a);
    
    // Initialize center (Mu) as midpoint of bounds
    for (j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = 0.5 * (opt->bounds[2 * j] + opt->bounds[2 * j + 1]);
    }
    opt->best_solution.fitness = INFINITY;

    // Initialize candidates with normal distribution around center
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            double radius = ginv * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) / 2.0;
            // Approximate normal distribution using Box-Muller transform
            double u1 = rand_double(0.0, 1.0);
            double u2 = rand_double(0.0, 1.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            opt->population[i].position[j] = opt->best_solution.position[j] + radius * z;
        }
    }
    enforce_bound_constraints(opt);
}

// First Phase: Generate candidate solutions around the vortex center
void first_phase(Optimizer *opt, int iteration, double radius) {
    int i, j, size;
    size = (iteration == 0) ? opt->population_size : opt->population_size / 2;

    for (i = 0; i < size; i++) {
        for (j = 0; j < opt->dim; j++) {
            double u1 = rand_double(0.0, 1.0);
            double u2 = rand_double(0.0, 1.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            opt->population[i].position[j] = opt->best_solution.position[j] + radius * z;
        }
    }
    enforce_bound_constraints(opt);
}

// Polynomial Mutation
void polynomial_mutation(Optimizer *opt, double *solution, double *mutated, int *state) {
    int i;
    double prob_mut = 1.0 / opt->dim;
    double mut_pow = 1.0 / (1.0 + DISTRIBUTION_INDEX);
    *state = 0;

    for (i = 0; i < opt->dim; i++) {
        mutated[i] = solution[i];
        if (rand_double(0.0, 1.0) <= prob_mut) {
            double y = solution[i];
            double yL = opt->bounds[2 * i];
            double yU = opt->bounds[2 * i + 1];
            double delta1 = (y - yL) / (yU - yL);
            double delta2 = (yU - y) / (yU - yL);
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

            y = y + deltaq * (yU - yL);
            if (y < yL) y = yL;
            if (y > yU) y = yU;
            mutated[i] = y;
            (*state)++;
        }
    }
}

// Second Phase: Crossover and Mutation
void second_phase(Optimizer *opt, int iteration, ObjectiveFunction objective_function) {
    int i, j, neighbor, param2change;
    double prob_cross = 1.0 / opt->dim;
    double *obj_vals = (double *)malloc(opt->population_size * sizeof(double));
    double *prob = (double *)malloc(opt->population_size * sizeof(double));
    double *sol = (double *)malloc(opt->dim * sizeof(double));
    double *mutated = (double *)malloc(opt->dim * sizeof(double));
    int state;

    // Evaluate all candidates
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
    for (i = 0; i < opt->population_size; i++) {
        prob[i] /= sum_prob;
    }
    for (i = 1; i < opt->population_size; i++) {
        prob[i] += prob[i - 1];
    }

    // Process second half of population
    for (i = opt->population_size / 2; i < opt->population_size; i++) {
        // Roulette wheel selection
        double r = rand_double(0.0, 1.0);
        neighbor = 0;
        for (j = 0; j < opt->population_size; j++) {
            if (r <= prob[j]) {
                neighbor = j;
                break;
            }
        }
        while (i == neighbor) {
            r = rand_double(0.0, 1.0);
            for (j = 0; j < opt->population_size; j++) {
                if (r <= prob[j]) {
                    neighbor = j;
                    break;
                }
            }
        }

        // Crossover
        memcpy(sol, opt->population[i].position, opt->dim * sizeof(double));
        param2change = rand() % opt->dim;
        for (j = 0; j < opt->dim; j++) {
            if (rand_double(0.0, 1.0) < prob_cross || j == param2change) {
                sol[j] += (opt->population[i].position[j] - opt->population[neighbor].position[j]) * (rand_double(0.0, 1.0) - 0.5) * 2.0;
            }
        }
        for (j = 0; j < opt->dim; j++) {
            if (sol[j] < opt->bounds[2 * j]) sol[j] = opt->bounds[2 * j];
            if (sol[j] > opt->bounds[2 * j + 1]) sol[j] = opt->bounds[2 * j + 1];
        }

        // Evaluate new solution
        double obj_val_sol = objective_function(sol);
        if (obj_val_sol < obj_vals[i]) {
            memcpy(opt->population[i].position, sol, opt->dim * sizeof(double));
            opt->population[i].fitness = obj_val_sol;
        } else {
            polynomial_mutation(opt, opt->population[i].position, mutated, &state);
            if (state > 0) {
                double obj_val_mut = objective_function(mutated);
                if (obj_val_mut < obj_vals[i]) {
                    memcpy(opt->population[i].position, mutated, opt->dim * sizeof(double));
                    opt->population[i].fitness = obj_val_mut;
                }
            }
        }
    }

    free(obj_vals);
    free(prob);
    free(sol);
    free(mutated);
}

// Main Optimization Function
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int iteration = 0;
    double x = X_GAMMA;
    double a, ginv;

    initialize_vortex(opt);

    while (iteration < opt->max_iter) {
        // Update radius
        a = (opt->max_iter - iteration) / (double)opt->max_iter;
        a = fmax(a, 0.1);
        ginv = (1.0 / x) * approx_gammaincinv(x, a);
        double radius = ginv * (opt->bounds[1] - opt->bounds[0]) / 2.0;

        // First phase
        first_phase(opt, iteration, radius);
        for (int i = 0; i < (iteration == 0 ? opt->population_size : opt->population_size / 2); i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Second phase
        second_phase(opt, iteration, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        iteration++;
    }
}
