/* PSO.c - Optimized Particle Swarm Optimization Implementation */
#include "PSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Fast random double between 0 and 1 (thread-safe)
static inline double fast_rand(double *seed) {
    *seed = *seed * 1103515245 + 12345;
    return ((unsigned)(*seed / 65536) % 32768) / 32768.0;
}

// Initialize Swarm (Contiguous Memory, Parallelized)
void PSO_initialize_swarm(Optimizer *opt, double *velocities, double *pbest_positions, double *pbest_fitnesses) {
    if (!opt || !velocities || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_initialize_swarm: Invalid arguments\n");
        return;
    }

    double seed = (double)time(NULL) + omp_get_thread_num();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < opt->population_size; i++) {
        pbest_fitnesses[i] = INFINITY;
        opt->population[i].fitness = INFINITY;

        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            int idx = i * opt->dim + j;
            opt->population[i].position[j] = min + fast_rand(&seed) * (max - min);
            velocities[idx] = 0.0;
            pbest_positions[idx] = opt->population[i].position[j];
        }
    }
    enforce_bound_constraints(opt);
}

// Update Velocity and Position (Contiguous Memory, Parallelized)
void PSO_update_velocity_position(Optimizer *opt, double *velocities, double *pbest_positions) {
    if (!opt || !velocities || !pbest_positions) {
        fprintf(stderr, "PSO_update_velocity_position: Invalid arguments\n");
        return;
    }

    static double w = INERTIA_WEIGHT;
    double seed = (double)time(NULL) + omp_get_thread_num();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            double vel_min = -VELOCITY_SCALE * (max - min);
            double vel_max = VELOCITY_SCALE * (max - min);
            int idx = i * opt->dim + j;

            double r1 = fast_rand(&seed);
            double r2 = fast_rand(&seed);

            velocities[idx] = w * velocities[idx] +
                PERSONAL_LEARNING * r1 * (pbest_positions[idx] - opt->population[i].position[j]) +
                GLOBAL_LEARNING * r2 * (opt->best_solution.position[j] - opt->population[i].position[j]);

            if (velocities[idx] < vel_min) {
                velocities[idx] = vel_min;
            } else if (velocities[idx] > vel_max) {
                velocities[idx] = vel_max;
            }

            opt->population[i].position[j] += velocities[idx];

            // Inline bounds enforcement
            if (opt->population[i].position[j] < min) {
                opt->population[i].position[j] = min;
                velocities[idx] = -velocities[idx];
            } else if (opt->population[i].position[j] > max) {
                opt->population[i].position[j] = max;
                velocities[idx] = -velocities[idx];
            }
        }
    }
    w *= INERTIA_DAMPING;
}

// Evaluate Particles and Update Bests (Parallelized)
void PSO_evaluate_particles(Optimizer *opt, double (*objective_function)(double *), 
                           double *pbest_positions, double *pbest_fitnesses) {
    if (!opt || !objective_function || !pbest_positions || !pbest_fitnesses) {
        fprintf(stderr, "PSO_evaluate_particles: Invalid arguments\n");
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;

        if (fitness < pbest_fitnesses[i]) {
            pbest_fitnesses[i] = fitness;
            for (int j = 0; j < opt->dim; j++) {
                pbest_positions[i * opt->dim + j] = opt->population[i].position[j];
            }
        }

        #pragma omp critical
        {
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
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

    // Contiguous memory allocation
    double *velocities = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *pbest_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
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

    // Cleanup
    free(velocities);
    free(pbest_positions);
    free(pbest_fitnesses);
}
