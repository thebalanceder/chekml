#include "BA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Frequency Update and Velocity Adjustment Phase
void bat_frequency_update(Optimizer *opt) {
    static double *freq = NULL;
    static double *velocities = NULL;
    static int initialized = 0;

    // Initialize frequency and velocity arrays (persistent across calls)
    if (!initialized) {
        freq = (double *)malloc(opt->population_size * sizeof(double));
        velocities = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
        for (int i = 0; i < opt->population_size * opt->dim; i++) {
            velocities[i] = 0.0;  // Initialize velocities to zero
        }
        initialized = 1;
    }

    for (int i = 0; i < opt->population_size; i++) {
        // Update frequency
        freq[i] = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * rand_double(0.0, 1.0);

        // Update velocity
        for (int j = 0; j < opt->dim; j++) {
            velocities[i * opt->dim + j] += (opt->population[i].position[j] - opt->best_solution.position[j]) * freq[i];
            // Update position
            opt->population[i].position[j] += velocities[i * opt->dim + j];
        }
    }

    enforce_bound_constraints(opt);
}

// Local Search Phase
void bat_local_search(Optimizer *opt) {
    static double loudness = LOUDNESS;
    static int iter = 0;

    // Update pulse rate and loudness
    double pulse_rate = PULSE_RATE * (1.0 - exp(-GAMMA * iter));
    loudness = ALPHA_BA * loudness;
    iter++;

    for (int i = 0; i < opt->population_size; i++) {
        if (rand_double(0.0, 1.0) < pulse_rate) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = opt->best_solution.position[j] + LOCAL_SEARCH_SCALE * (2.0 * rand_double(0.0, 1.0) - 1.0) * loudness;
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Solution Update Phase (Evaluate and Update Best)
void bat_update_solutions(Optimizer *opt, double (*objective_function)(double *)) {
    static double loudness = LOUDNESS;

    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        // Accept new solution if it improves or not too loud
        if (new_fitness <= opt->population[i].fitness && rand_double(0.0, 1.0) > loudness) {
            opt->population[i].fitness = new_fitness;
        }
        // Update global best
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    loudness = ALPHA_BA * loudness;  // Update loudness for next iteration
}

// Main Optimization Function
void BA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize population fitness
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        bat_frequency_update(opt);
        bat_local_search(opt);
        bat_update_solutions(opt, objective_function);

        // Debugging output every 100 iterations
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }
}
