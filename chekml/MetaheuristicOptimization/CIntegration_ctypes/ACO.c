#include "ACO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

double rand_double(double min, double max);

// Roulette wheel selection
int roulette_wheel_selection(double *P, int size) {
    double r = rand_double(0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += P[i];
        if (r <= cumsum) {
            return i;
        }
    }
    return size - 1; // Fallback
}

// Construct solutions for all ants
void construct_solutions(Optimizer *opt, double (*objective_function)(double *), double **tau) {
    double *P = (double *)malloc(ACO_N_BINS * sizeof(double));
    double **bins = (double **)malloc(ACO_N_BINS * sizeof(double *));
    double **eta = (double **)malloc(ACO_N_BINS * sizeof(double *));
    int **tours = (int **)malloc(opt->population_size * sizeof(int *));
    double sum_P;

    // Allocate memory for bins, eta, and tours
    for (int i = 0; i < ACO_N_BINS; i++) {
        bins[i] = (double *)malloc(opt->dim * sizeof(double));
        eta[i] = (double *)malloc(opt->dim * sizeof(double));
    }
    for (int i = 0; i < opt->population_size; i++) {
        tours[i] = (int *)malloc(opt->dim * sizeof(int));
    }

    // Initialize bins and eta
    for (int d = 0; d < opt->dim; d++) {
        for (int i = 0; i < ACO_N_BINS; i++) {
            bins[i][d] = opt->bounds[2 * d] + (opt->bounds[2 * d + 1] - opt->bounds[2 * d]) * i / (ACO_N_BINS - 1);
            eta[i][d] = 1.0; // Uniform heuristic
        }
    }

    // Construct solutions
    for (int k = 0; k < opt->population_size; k++) {
        for (int d = 0; d < opt->dim; d++) {
            // Compute probabilities
            sum_P = 0.0;
            for (int i = 0; i < ACO_N_BINS; i++) {
                P[i] = pow(tau[i][d], ACO_ALPHA) * pow(eta[i][d], ACO_BETA);
                sum_P += P[i];
            }

            // Normalize probabilities
            if (sum_P > 0.0) {
                for (int i = 0; i < ACO_N_BINS; i++) {
                    P[i] /= sum_P;
                }
            }

            // Select bin
            int bin_idx = roulette_wheel_selection(P, ACO_N_BINS);
            tours[k][d] = bin_idx;
            opt->population[k].position[d] = bins[bin_idx][d];
        }

        // Evaluate fitness
        opt->population[k].fitness = objective_function(opt->population[k].position);
    }

    // Free memory
    for (int i = 0; i < ACO_N_BINS; i++) {
        free(bins[i]);
        free(eta[i]);
    }
    for (int i = 0; i < opt->population_size; i++) {
        free(tours[i]);
    }
    free(bins);
    free(eta);
    free(tours);
    free(P);

    enforce_bound_constraints(opt);
}

// Update pheromone trails
void update_pheromones(Optimizer *opt, double **tau) {
    for (int k = 0; k < opt->population_size; k++) {
        for (int d = 0; d < opt->dim; d++) {
            int bin_idx = (int)(ACO_N_BINS * (opt->population[k].position[d] - opt->bounds[2 * d]) / 
                               (opt->bounds[2 * d + 1] - opt->bounds[2 * d]));
            if (bin_idx >= ACO_N_BINS) bin_idx = ACO_N_BINS - 1;
            if (bin_idx < 0) bin_idx = 0;
            tau[bin_idx][d] += ACO_Q / (1.0 + fmax(0.0, opt->population[k].fitness - opt->best_solution.fitness));
        }
    }
}

// Apply pheromone evaporation
void evaporate_pheromones(double **tau, int dim) {
    for (int i = 0; i < ACO_N_BINS; i++) {
        for (int d = 0; d < dim; d++) {
            tau[i][d] *= (1.0 - ACO_RHO);
        }
    }
}

// Main Optimization Function
void ACO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL)); // Seed random number generator

    // Allocate persistent pheromone matrix
    double **tau = (double **)malloc(ACO_N_BINS * sizeof(double *));
    for (int i = 0; i < ACO_N_BINS; i++) {
        tau[i] = (double *)malloc(opt->dim * sizeof(double));
        for (int d = 0; d < opt->dim; d++) {
            tau[i][d] = ACO_TAU0; // Initialize pheromones
        }
    }

    // Set population size and max iterations
    opt->population_size = ACO_N_ANT;
    opt->max_iter = ACO_MAX_ITER;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        construct_solutions(opt, objective_function, tau);
        update_pheromones(opt, tau);
        evaporate_pheromones(tau, opt->dim);

        // Find best solution
        double current_best_fitness = INFINITY;
        int current_best_idx = 0;
        for (int k = 0; k < opt->population_size; k++) {
            if (opt->population[k].fitness < current_best_fitness) {
                current_best_fitness = opt->population[k].fitness;
                current_best_idx = k;
            }
        }

        // Update global best
        if (current_best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = current_best_fitness;
            for (int d = 0; d < opt->dim; d++) {
                opt->best_solution.position[d] = opt->population[current_best_idx].position[d];
            }
        }

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free pheromone matrix
    for (int i = 0; i < ACO_N_BINS; i++) {
        free(tau[i]);
    }
    free(tau);
}
