/* PSO.c - Implementation file for Particle Swarm Optimization */
#include "PSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Swarm (Random Positions and Velocities)
void PSO_initialize_swarm(Optimizer *opt, double **velocities, double **pbest_positions, double *pbest_fitnesses) {
    if (!opt || !velocities || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_initialize_swarm: Invalid arguments\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        velocities[i] = (double *)malloc(opt->dim * sizeof(double));
        pbest_positions[i] = (double *)malloc(opt->dim * sizeof(double));
        if (!velocities[i] || !pbest_positions[i]) {
            fprintf(stderr, "PSO_initialize_swarm: Memory allocation failed\n");
            exit(1);
        }
        pbest_fitnesses[i] = INFINITY;
        opt->population[i].fitness = INFINITY;

        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = rand_double(min, max);
            velocities[i][j] = 0.0;
            pbest_positions[i][j] = opt->population[i].position[j];
        }
    }
    enforce_bound_constraints(opt);
}

// Update Velocity and Position
void PSO_update_velocity_position(Optimizer *opt, double **velocities, double **pbest_positions) {
    if (!opt || !velocities || !pbest_positions) {
        fprintf(stderr, "PSO_update_velocity_position: Invalid arguments\n");
        return;
    }

    static double w = INERTIA_WEIGHT;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            double vel_min = -VELOCITY_SCALE * (max - min);
            double vel_max = VELOCITY_SCALE * (max - min);

            double r1 = rand_double(0.0, 1.0);
            double r2 = rand_double(0.0, 1.0);

            velocities[i][j] = w * velocities[i][j] +
                PERSONAL_LEARNING * r1 * (pbest_positions[i][j] - opt->population[i].position[j]) +
                GLOBAL_LEARNING * r2 * (opt->best_solution.position[j] - opt->population[i].position[j]);

            if (velocities[i][j] < vel_min) {
                velocities[i][j] = vel_min;
            } else if (velocities[i][j] > vel_max) {
                velocities[i][j] = vel_max;
            }

            opt->population[i].position[j] += velocities[i][j];

            if (opt->population[i].position[j] < min || opt->population[i].position[j] > max) {
                velocities[i][j] = -velocities[i][j];
            }
        }
    }
    enforce_bound_constraints(opt);
    w *= INERTIA_DAMPING;
}

// Evaluate Particles and Update Bests
void PSO_evaluate_particles(Optimizer *opt, double (*objective_function)(double *), 
                           double **pbest_positions, double *pbest_fitnesses) {
    if (!opt || !objective_function || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_evaluate_particles: Invalid arguments\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;

        if (fitness < pbest_fitnesses[i]) {
            pbest_fitnesses[i] = fitness;
            for (int j = 0; j < opt->dim; j++) {
                pbest_positions[i][j] = opt->population[i].position[j];
            }
        }

        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Main Optimization Function
void PSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "PSO_optimize: Invalid arguments\n");
        return;
    }

    double **velocities = (double **)malloc(opt->population_size * sizeof(double *));
    double **pbest_positions = (double **)malloc(opt->population_size * sizeof(double *));
    double *pbest_fitnesses = (double *)malloc(opt->population_size * sizeof(double));

    if (!velocities || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_optimize: Memory allocation failed\n");
        free(velocities);
        free(pbest_positions);
        free(pbest_fitnesses);
        return;
    }

    PSO_initialize_swarm(opt, velocities, pbest_positions, pbest_fitnesses);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        PSO_update_velocity_position(opt, velocities, pbest_positions);
        PSO_evaluate_particles(opt, objective_function, pbest_positions, pbest_fitnesses);
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Cleanup PSO-specific arrays
    for (int i = 0; i < opt->population_size; i++) {
        free(velocities[i]);
        free(pbest_positions[i]);
    }
    free(velocities);
    free(pbest_positions);
    free(pbest_fitnesses);
}
