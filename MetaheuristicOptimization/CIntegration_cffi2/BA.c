#include "BA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// 🦇 Frequency Update and Velocity Adjustment Phase
void bat_frequency_update(Optimizer *restrict opt, double *restrict freq, double *restrict velocities) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double freq_range = FREQ_MAX - FREQ_MIN;

    for (int i = 0; i < pop_size; i++) {
        // Update frequency
        freq[i] = FREQ_MIN + freq_range * ((double)rand() / RAND_MAX);

        // Update velocity and position
        double *pos = opt->population[i].position;
        double *vel = velocities + i * dim;
        const double *best_pos = opt->best_solution.position;
        const double freq_i = freq[i];

        for (int j = 0; j < dim; j++) {
            vel[j] += (pos[j] - best_pos[j]) * freq_i;
            pos[j] += vel[j];
        }
    }

    enforce_bound_constraints(opt);
}

// 🦇 Local Search Phase
void bat_local_search(Optimizer *restrict opt, double *restrict freq, double pulse_rate, double loudness) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double scale = LOCAL_SEARCH_SCALE * loudness;

    for (int i = 0; i < pop_size; i++) {
        if (((double)rand() / RAND_MAX) < pulse_rate) {
            double *pos = opt->population[i].position;
            const double *best_pos = opt->best_solution.position;
            for (int j = 0; j < dim; j++) {
                pos[j] = best_pos[j] + scale * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
            }
        }
    }

    enforce_bound_constraints(opt);
}

// 🦇 Solution Update Phase (Evaluate and Update Best)
void bat_update_solutions(Optimizer *restrict opt, double *restrict freq, double loudness, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;

    for (int i = 0; i < pop_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        // Accept new solution if it improves or not too loud
        if (new_fitness <= opt->population[i].fitness && ((double)rand() / RAND_MAX) > loudness) {
            opt->population[i].fitness = new_fitness;
        }
        // Update global best
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            double *pos = opt->population[i].position;
            double *best_pos = opt->best_solution.position;
            for (int j = 0; j < dim; j++) {
                best_pos[j] = pos[j];
            }
        }
    }
}

// 🚀 Main Optimization Function
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Allocate frequency and velocity arrays
    double *freq = (double *)malloc(opt->population_size * sizeof(double));
    double *velocities = (double *)calloc(opt->population_size * opt->dim, sizeof(double)); // Zero-initialized
    if (!freq || !velocities) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize population fitness and find initial best
    int pop_size = opt->population_size;
    int dim = opt->dim;
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            double *pos = opt->population[i].position;
            double *best_pos = opt->best_solution.position;
            for (int j = 0; j < dim; j++) {
                best_pos[j] = pos[j];
            }
        }
    }

    // Main optimization loop
    double loudness = LOUDNESS;
    double pulse_rate = PULSE_RATE;
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update pulse rate and loudness
        pulse_rate = PULSE_RATE * (1.0 - exp(-GAMMA * iter));
        loudness *= ALPHA_BA;

        bat_frequency_update(opt, freq, velocities);
        bat_local_search(opt, freq, pulse_rate, loudness);
        bat_update_solutions(opt, freq, loudness, objective_function);

        // Debugging output every 100 iterations
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
        }
    }

    // Clean up
    free(freq);
    free(velocities);
}
