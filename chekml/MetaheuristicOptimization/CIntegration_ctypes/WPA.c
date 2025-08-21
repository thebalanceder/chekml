/* WPA.c - Implementation file for Waterwheel Plant Algorithm (WWPA) */
#include "WPA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Gaussian random number generator (simplified using Box-Muller transform)
double gaussian(double mu, double sigma) {
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    return mu + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// Exploration Phase (Position Identification and Hunting)
void wpa_exploration_phase(Optimizer *opt, int t, double K, int *stagnation_counts, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double r1 = rand_double(R1_MIN, R1_MAX);
        double r2 = rand_double(R2_MIN, R2_MAX);
        double *W = (double *)malloc(opt->dim * sizeof(double));
        double *new_position = (double *)malloc(opt->dim * sizeof(double));
        double new_value, current_value;

        // Equation (4): W = r1 * (P(t) + 2K)
        for (int j = 0; j < opt->dim; j++) {
            W[j] = r1 * (opt->population[i].position[j] + 2.0 * K);
        }

        // Equation (5): P(t+1) = P(t) + W * (2K + r2)
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + W[j] * (2.0 * K + r2);
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }

        // Evaluate fitness
        new_value = objective_function(new_position);
        current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_position[j];
            }
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
                double *mu_P = (double *)calloc(opt->dim, sizeof(double));
                double *sigma = (double *)calloc(opt->dim, sizeof(double));

                // Compute mean and standard deviation
                for (int j = 0; j < opt->dim; j++) {
                    for (int k = 0; k < opt->population_size; k++) {
                        mu_P[j] += opt->population[k].position[j];
                    }
                    mu_P[j] /= opt->population_size;
                }
                for (int j = 0; j < opt->dim; j++) {
                    for (int k = 0; k < opt->population_size; k++) {
                        sigma[j] += pow(opt->population[k].position[j] - mu_P[j], 2);
                    }
                    sigma[j] = sqrt(sigma[j] / opt->population_size);
                }

                for (int j = 0; j < opt->dim; j++) {
                    double gaussian_term = gaussian(mu_P[j], sigma[j]);
                    if (W[j] != 0.0) {
                        new_position[j] = gaussian_term + r1 * (opt->population[i].position[j] + 2.0 * K) / W[j];
                    } else {
                        new_position[j] = gaussian_term;
                    }
                    new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = new_position[j];
                }
                opt->population[i].fitness = objective_function(new_position);

                free(mu_P);
                free(sigma);
            }
        }

        free(W);
        free(new_position);
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase (Carrying Insect to Suitable Tube)
void wpa_exploitation_phase(Optimizer *opt, int t, double K, int *stagnation_counts, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double r3 = rand_double(R3_MIN, R3_MAX);
        double *W = (double *)malloc(opt->dim * sizeof(double));
        double *new_position = (double *)malloc(opt->dim * sizeof(double));
        double new_value, current_value;

        // Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
        for (int j = 0; j < opt->dim; j++) {
            W[j] = r3 * (K * opt->best_solution.position[j] + r3 * opt->population[i].position[j]);
        }

        // Equation (8): P(t+1) = P(t) + K * W
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + K * W[j];
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }

        // Evaluate fitness
        new_value = objective_function(new_position);
        current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_position[j];
            }
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
                double r1 = rand_double(R1_MIN, R1_MAX);
                double f = rand_double(F_MIN, F_MAX);
                double c = rand_double(C_MIN, C_MAX);
                double theta = rand_double(0.0, 1.0) * 2.0 * PI;

                for (int j = 0; j < opt->dim; j++) {
                    if (c != 0.0) {
                        new_position[j] = (r1 + K) * sin((f / c) * theta);
                    } else {
                        new_position[j] = opt->population[i].position[j];
                    }
                    new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = new_position[j];
                }
                opt->population[i].fitness = objective_function(new_position);
            }
        }

        free(W);
        free(new_position);
    }
    enforce_bound_constraints(opt);
}

// Update K Parameter (Equation 10)
double wpa_update_k(int t, int max_iter, double f) {
    return 1.0 + (2.0 * t * t / (max_iter * max_iter)) + f;
}

// Main Optimization Function
void WPA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int *stagnation_counts = (int *)calloc(opt->population_size, sizeof(int));

    for (int iter = 0; iter < opt->max_iter; iter++) {
        double r = rand_double(0.0, 1.0);
        double f = rand_double(F_MIN, F_MAX);
        double K = wpa_update_k(iter, opt->max_iter, f);

        if (r < 0.5) {
            wpa_exploration_phase(opt, iter, K, stagnation_counts, objective_function);
        } else {
            wpa_exploitation_phase(opt, iter, K, stagnation_counts, objective_function);
        }

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(stagnation_counts);
}
