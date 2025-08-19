/* LSA.c - Implementation file for Lightning Search Algorithm */
#include "LSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <math.h>    // For exp, fabs, etc.

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a random normal distribution (approximation using Box-Muller)
double rand_normal_lsa(double mean, double stddev) {
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z;
}

// Function to generate an exponential distribution
double rand_exponential(double lambda) {
    double u = rand_double(0.0, 1.0);
    return -log(1.0 - u) / lambda;
}

// Initialize Channels
void initialize_channels(Optimizer *opt, double *directions, int *channel_time) {
    fprintf(stderr, "initialize_channels: Initializing %d channels\n", opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
    }
    // Initialize directions
    for (int j = 0; j < opt->dim; j++) {
        directions[j] = rand_double(-1.0, 1.0);
    }
    *channel_time = 0;
    enforce_bound_constraints(opt);
}

// Evaluate Channels
void evaluate_channels(Optimizer *opt, ObjectiveFunction objective_function) {
    fprintf(stderr, "evaluate_channels: Evaluating %d channels\n", opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Update Channel Elimination
void update_channel_elimination(Optimizer *opt, double *directions, int *channel_time) {
    fprintf(stderr, "update_channel_elimination: channel_time=%d\n", *channel_time);
    (*channel_time)++;
    if (*channel_time >= MAX_CHANNEL_TIME) {
        // Find best and worst channels
        int best_idx = 0, worst_idx = 0;
        double best_fitness = opt->population[0].fitness;
        double worst_fitness = opt->population[0].fitness;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < best_fitness) {
                best_fitness = opt->population[i].fitness;
                best_idx = i;
            }
            if (opt->population[i].fitness > worst_fitness) {
                worst_fitness = opt->population[i].fitness;
                worst_idx = i;
            }
        }
        // Replace worst with best
        for (int j = 0; j < opt->dim; j++) {
            opt->population[worst_idx].position[j] = opt->population[best_idx].position[j];
        }
        opt->population[worst_idx].fitness = opt->population[best_idx].fitness;
        *channel_time = 0;
    }
}

// Update Directions
void update_directions(Optimizer *opt, double *directions, ObjectiveFunction objective_function) {
    fprintf(stderr, "update_directions: Updating directions\n");
    int best_idx = 0;
    double best_fitness = opt->population[0].fitness;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    double *test_channel = (double *)malloc(opt->dim * sizeof(double));
    if (!test_channel) {
        fprintf(stderr, "update_directions: Memory allocation failed\n");
        return;
    }

    for (int j = 0; j < opt->dim; j++) {
        test_channel[j] = opt->population[best_idx].position[j];
    }

    for (int j = 0; j < opt->dim; j++) {
        double lb = opt->bounds[2 * j];
        double ub = opt->bounds[2 * j + 1];
        test_channel[j] += directions[j] * DIRECTION_STEP * (ub - lb);
        double test_fitness = objective_function(test_channel);
        if (test_fitness < best_fitness) {
            directions[j] = directions[j];
        } else {
            directions[j] = -directions[j];
        }
        test_channel[j] = opt->population[best_idx].position[j]; // Reset
    }

    free(test_channel);
}

// Update Positions
void lsa_update_positions(Optimizer *opt, double *directions, ObjectiveFunction objective_function, int t) {
    fprintf(stderr, "lsa_update_positions: Iteration %d\n", t);
    double energy = LSA_ENERGY_FACTOR - 2.0 * exp(-5.0 * (opt->max_iter - t) / (double)opt->max_iter);
    int best_idx = 0;
    double best_fitness = opt->population[0].fitness;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    double *temp_channel = (double *)malloc(opt->dim * sizeof(double));
    double *fock_channel = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_channel || !fock_channel) {
        fprintf(stderr, "lsa_update_positions: Memory allocation failed\n");
        free(temp_channel);
        free(fock_channel);
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double is_best = 1.0;
        for (int j = 0; j < opt->dim; j++) {
            if (fabs(opt->population[i].position[j] - opt->population[best_idx].position[j]) > 1e-10) {
                is_best = 0.0;
                break;
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            if (is_best) {
                temp_channel[j] = opt->population[i].position[j] + directions[j] * fabs(rand_normal_lsa(0.0, energy));
            } else {
                double dist = opt->population[i].position[j] - opt->population[best_idx].position[j];
                double r = rand_exponential(fabs(dist));
                temp_channel[j] = opt->population[i].position[j] + (dist < 0 ? r : -r);
            }

            // Boundary check
            if (temp_channel[j] < lb || temp_channel[j] > ub) {
                temp_channel[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
            }
        }

        double temp_fitness = objective_function(temp_channel);
        if (temp_fitness < opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = temp_channel[j];
            }
            opt->population[i].fitness = temp_fitness;

            // Focking procedure
            if (rand_double(0.0, 1.0) < FOCKING_PROB) {
                for (int j = 0; j < opt->dim; j++) {
                    double lb = opt->bounds[2 * j];
                    double ub = opt->bounds[2 * j + 1];
                    fock_channel[j] = lb + ub - temp_channel[j];
                }
                double fock_fitness = objective_function(fock_channel);
                if (fock_fitness < opt->population[i].fitness) {
                    for (int j = 0; j < opt->dim; j++) {
                        opt->population[i].position[j] = fock_channel[j];
                    }
                    opt->population[i].fitness = fock_fitness;
                }
            }
        }
    }

    free(temp_channel);
    free(fock_channel);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void LSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    fprintf(stderr, "LSA_optimize: Starting optimization\n");
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Allocate directions
    double *directions = (double *)malloc(opt->dim * sizeof(double));
    int channel_time = 0;
    if (!directions) {
        fprintf(stderr, "LSA_optimize: Memory allocation failed for directions\n");
        return;
    }
    fprintf(stderr, "LSA_optimize: Allocated directions at %p\n", directions);

    // Initialize channels
    initialize_channels(opt, directions, &channel_time);

    for (int t = 0; t < opt->max_iter; t++) {
        // Evaluate channels
        evaluate_channels(opt, objective_function);

        // Update channel elimination
        update_channel_elimination(opt, directions, &channel_time);

        // Update directions
        update_directions(opt, directions, objective_function);

        // Update positions
        lsa_update_positions(opt, directions, objective_function, t);

        // Check for convergence
        double best_fitness = opt->population[0].fitness;
        double worst_fitness = opt->population[0].fitness;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < best_fitness) {
                best_fitness = opt->population[i].fitness;
            }
            if (opt->population[i].fitness > worst_fitness) {
                worst_fitness = opt->population[i].fitness;
            }
        }
        if (fabs(best_fitness - worst_fitness) < 1e-10) {
            fprintf(stderr, "LSA_optimize: Converged at iteration %d\n", t);
            break;
        }
    }

    // Final evaluation to ensure best solution is up-to-date
    evaluate_channels(opt, objective_function);

    // Free directions
    fprintf(stderr, "LSA_optimize: Freeing directions at %p\n", directions);
    free(directions);
    fprintf(stderr, "LSA_optimize: Optimization completed\n");
}
