#include "ACO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
inline double rand_double_aco(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Roulette wheel selection
inline int roulette_wheel_selection(double *prob, int size) {
    double r = rand_double_aco(0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += prob[i];
        if (r <= cumsum) {
            return i;
        }
    }
    return size - 1; // Fallback
}

// Construct solutions for all ants
void construct_solutions(Optimizer *opt, double (*objective_function)(double *), double **tau, double **bins, int **tours, double *prob) {
    double sum_P;

    // Construct solutions
    for (int k = 0; k < opt->population_size; k++) {
        for (int d = 0; d < opt->dim; d++) {
            // Compute probabilities (simplified since eta=1.0 and ACO_ALPHA=1.0)
            sum_P = 0.0;
            for (int i = 0; i < ACO_N_BINS; i++) {
                prob[i] = tau[i][d]; // eta[i][d] = 1.0, pow(1.0, ACO_BETA) = 1.0, pow(tau, 1.0) = tau
                sum_P += prob[i];
            }

            // Normalize probabilities
            if (sum_P > 0.0) {
                double inv_sum_P = 1.0 / sum_P;
                for (int i = 0; i < ACO_N_BINS; i++) {
                    prob[i] *= inv_sum_P;
                }
            }

            // Select bin
            int bin_idx = roulette_wheel_selection(prob, ACO_N_BINS);
            tours[k][d] = bin_idx;
            opt->population[k].position[d] = bins[bin_idx][d];
        }

        // Evaluate fitness
        opt->population[k].fitness = objective_function(opt->population[k].position);
    }

    enforce_bound_constraints(opt);
}

// Update pheromone trails using precomputed tours
void update_pheromones(Optimizer *opt, double **tau, int **tours) {
    for (int k = 0; k < opt->population_size; k++) {
        double delta = ACO_Q / (1.0 + (opt->population[k].fitness > opt->best_solution.fitness ? 
                                      opt->population[k].fitness - opt->best_solution.fitness : 0.0));
        for (int d = 0; d < opt->dim; d++) {
            tau[tours[k][d]][d] += delta;
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

    // Allocate persistent arrays
    double **tau = (double **)malloc(ACO_N_BINS * sizeof(double *));
    double **bins = (double **)malloc(ACO_N_BINS * sizeof(double *));
    int **tours = (int **)malloc(ACO_N_ANT * sizeof(int *));
    double *prob = (double *)malloc(ACO_N_BINS * sizeof(double));

    for (int i = 0; i < ACO_N_BINS; i++) {
        tau[i] = (double *)malloc(opt->dim * sizeof(double));
        bins[i] = (double *)malloc(opt->dim * sizeof(double));
    }
    for (int i = 0; i < ACO_N_ANT; i++) {
        tours[i] = (int *)malloc(opt->dim * sizeof(int));
    }

    // Initialize tau and bins
    for (int d = 0; d < opt->dim; d++) {
        double lower = opt->bounds[2 * d];
        double range = opt->bounds[2 * d + 1] - lower;
        double bin_step = range / (ACO_N_BINS - 1);
        for (int i = 0; i < ACO_N_BINS; i++) {
            tau[i][d] = ACO_TAU0;
            bins[i][d] = lower + i * bin_step;
        }
    }

    // Set population size and max iterations
    opt->population_size = ACO_N_ANT;
    opt->max_iter = ACO_MAX_ITER;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        construct_solutions(opt, objective_function, tau, bins, tours, prob);
        update_pheromones(opt, tau, tours);
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

    // Free allocated memory
    for (int i = 0; i < ACO_N_BINS; i++) {
        free(tau[i]);
        free(bins[i]);
    }
    for (int i = 0; i < ACO_N_ANT; i++) {
        free(tours[i]);
    }
    free(tau);
    free(bins);
    free(tours);
    free(prob);
}
