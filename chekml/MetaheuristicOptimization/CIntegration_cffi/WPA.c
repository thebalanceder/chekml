/* WPA.c - Implementation file for Waterwheel Plant Algorithm (WWPA) */
#include "WPA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Gaussian random number generator (simplified using Box-Muller transform)
static inline double gaussian(double mu, double sigma) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return mu + sigma * spare;
    }
    has_spare = 1;
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * PI * u2;
    spare = r * sin(theta);
    return mu + sigma * (r * cos(theta));
}

// Compute population mean and standard deviation
static void compute_stats(Optimizer *restrict opt, double *restrict mu_P, double *restrict sigma) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *pop_pos;
    double sum, sum_sq;

    for (int j = 0; j < dim; j++) {
        sum = 0.0;
        sum_sq = 0.0;
        for (int i = 0; i < pop_size; i++) {
            pop_pos = opt->population[i].position;
            sum += pop_pos[j];
            sum_sq += pop_pos[j] * pop_pos[j];
        }
        mu_P[j] = sum / pop_size;
        sigma[j] = sqrt((sum_sq / pop_size) - (mu_P[j] * mu_P[j]));
    }
}

// Exploration Phase (Position Identification and Hunting)
void wpa_exploration_phase(Optimizer *restrict opt, int t, double K, int *restrict stagnation_counts, 
                          double *restrict mu_P, double *restrict sigma, double (*objective_function)(double *)) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *W = (double *)malloc(dim * sizeof(double));
    double *new_position = (double *)malloc(dim * sizeof(double));
    double *pop_pos, *bounds;

    for (int i = 0; i < pop_size; i++) {
        double r1 = rand_double(R1_MIN, R1_MAX);
        double r2 = rand_double(R2_MIN, R2_MAX);
        pop_pos = opt->population[i].position;
        bounds = opt->bounds;

        // Equation (4): W = r1 * (P(t) + 2K)
        for (int j = 0; j < dim; j++) {
            W[j] = r1 * (pop_pos[j] + 2.0 * K);
        }

        // Equation (5): P(t+1) = P(t) + W * (2K + r2)
        double factor = 2.0 * K + r2;
        for (int j = 0; j < dim; j++) {
            new_position[j] = pop_pos[j] + W[j] * factor;
            new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
        }

        // Evaluate fitness
        double new_value = objective_function(new_position);
        double current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            memcpy(pop_pos, new_position, dim * sizeof(double));
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
                for (int j = 0; j < dim; j++) {
                    double gaussian_term = gaussian(mu_P[j], sigma[j]);
                    new_position[j] = (W[j] != 0.0) ? gaussian_term + r1 * (pop_pos[j] + 2.0 * K) / W[j] : gaussian_term;
                    new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                memcpy(pop_pos, new_position, dim * sizeof(double));
                opt->population[i].fitness = objective_function(new_position);
            }
        }
    }

    free(W);
    free(new_position);
    enforce_bound_constraints(opt);
}

// Exploitation Phase (Carrying Insect to Suitable Tube)
void wpa_exploitation_phase(Optimizer *restrict opt, int t, double K, int *restrict stagnation_counts, 
                           double (*objective_function)(double *)) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *W = (double *)malloc(dim * sizeof(double));
    double *new_position = (double *)malloc(dim * sizeof(double));
    double *pop_pos, *best_pos, *bounds;

    for (int i = 0; i < pop_size; i++) {
        double r3 = rand_double(R3_MIN, R3_MAX);
        pop_pos = opt->population[i].position;
        best_pos = opt->best_solution.position;
        bounds = opt->bounds;

        // Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
        for (int j = 0; j < dim; j++) {
            W[j] = r3 * (K * best_pos[j] + r3 * pop_pos[j]);
        }

        // Equation (8): P(t+1) = P(t) + K * W
        for (int j = 0; j < dim; j++) {
            new_position[j] = pop_pos[j] + K * W[j];
            new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
        }

        // Evaluate fitness
        double new_value = objective_function(new_position);
        double current_value = opt->population[i].fitness;

        if (new_value < current_value) {
            stagnation_counts[i] = 0;
            memcpy(pop_pos, new_position, dim * sizeof(double));
            opt->population[i].fitness = new_value;
        } else {
            stagnation_counts[i]++;
            if (stagnation_counts[i] >= STAGNATION_THRESHOLD) {
                // Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
                double r1 = rand_double(R1_MIN, R1_MAX);
                double f = rand_double(F_MIN, F_MAX);
                double c = rand_double(C_MIN, C_MAX);
                double theta = rand_double(0.0, 1.0) * 2.0 * PI;
                double factor = (c != 0.0) ? (r1 + K) * sin((f / c) * theta) : pop_pos[0];

                for (int j = 0; j < dim; j++) {
                    new_position[j] = factor;
                    new_position[j] = fmin(fmax(new_position[j], bounds[2 * j]), bounds[2 * j + 1]);
                }

                stagnation_counts[i] = 0;
                memcpy(pop_pos, new_position, dim * sizeof(double));
                opt->population[i].fitness = objective_function(new_position);
            }
        }
    }

    free(W);
    free(new_position);
    enforce_bound_constraints(opt);
}

// Update K Parameter (Equation 10)
static inline double wpa_update_k(int t, int max_iter, double f) {
    return 1.0 + (2.0 * t * t / (max_iter * max_iter)) + f;
}

// Main Optimization Function
void WPA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    int *stagnation_counts = (int *)calloc(pop_size, sizeof(int));
    double *mu_P = (double *)calloc(dim, sizeof(double));
    double *sigma = (double *)calloc(dim, sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        double r = rand_double(0.0, 1.0);
        double f = rand_double(F_MIN, F_MAX);
        double K = wpa_update_k(iter, max_iter, f);

        // Compute stats for exploration phase
        if (r < 0.5) {
            compute_stats(opt, mu_P, sigma);
            wpa_exploration_phase(opt, iter, K, stagnation_counts, mu_P, sigma, objective_function);
        } else {
            wpa_exploitation_phase(opt, iter, K, stagnation_counts, objective_function);
        }

        // Update best solution
        for (int i = 0; i < pop_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, dim * sizeof(double));
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(stagnation_counts);
    free(mu_P);
    free(sigma);
}
