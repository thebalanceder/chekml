
/* FA.c - Implementation file for Fireworks Algorithm (FWA) Optimization */
#include "FA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <math.h>    // For sqrt(), log(), cos(), M_PI

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a Gaussian random number (Box-Muller transform)
double rand_gaussian_fa() {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Initialize particle positions randomly within bounds
void initialize_particles(Optimizer *opt) {
    if (!opt || !opt->population) {
        fprintf(stderr, "🚫 Error: Invalid optimizer or population\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double min_bound = opt->bounds[2 * j];
            double max_bound = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = rand_double(min_bound, max_bound);
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all particles
void evaluate_fitness(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs in evaluate_fitness\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Generate sparks around the best particle
void generate_sparks(Optimizer *opt, double *best_particle, double *sparks) {
    if (!opt || !best_particle || !sparks) {
        fprintf(stderr, "🚫 Error: Invalid inputs in generate_sparks\n");
        return;
    }

    int num_sparks = (int)BETA;  // Fixed number of sparks
    for (int i = 0; i < num_sparks; i++) {
        for (int j = 0; j < opt->dim; j++) {
            sparks[i * opt->dim + j] = best_particle[j] + ALPHA * rand_gaussian_fa() * DELTA_T;
        }
    }
}

// Update particles by combining with sparks and selecting top performers
void update_particles(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs in update_particles\n");
        return;
    }

    // Find best particle
    int min_idx = 0;
    double min_fitness = opt->population[0].fitness;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < min_fitness) {
            min_fitness = opt->population[i].fitness;
            min_idx = i;
        }
    }

    // Update best solution if improved
    if (min_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = min_fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[min_idx].position[j];
        }
    }

    // Generate sparks
    int num_sparks = (int)BETA;
    double *sparks = (double *)malloc(num_sparks * opt->dim * sizeof(double));
    if (!sparks) {
        fprintf(stderr, "🚫 Error: Memory allocation failed for sparks\n");
        return;
    }
    generate_sparks(opt, opt->population[min_idx].position, sparks);

    // Combine particles and sparks
    int total_particles = opt->population_size + num_sparks;
    double *all_positions = (double *)malloc(total_particles * opt->dim * sizeof(double));
    double *all_fitness = (double *)malloc(total_particles * sizeof(double));
    if (!all_positions || !all_fitness) {
        fprintf(stderr, "🚫 Error: Memory allocation failed for all_particles\n");
        free(sparks);
        if (all_positions) free(all_positions);
        if (all_fitness) free(all_fitness);
        return;
    }

    // Copy particles
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            all_positions[i * opt->dim + j] = opt->population[i].position[j];
        }
        all_fitness[i] = opt->population[i].fitness;
    }

    // Copy sparks and evaluate fitness
    for (int i = 0; i < num_sparks; i++) {
        for (int j = 0; j < opt->dim; j++) {
            all_positions[(opt->population_size + i) * opt->dim + j] = sparks[i * opt->dim + j];
        }
        all_fitness[opt->population_size + i] = objective_function(&all_positions[(opt->population_size + i) * opt->dim]);
    }

    // Apply boundary constraints
    for (int i = 0; i < total_particles; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double val = all_positions[i * opt->dim + j];
            if (val < opt->bounds[2 * j]) {
                all_positions[i * opt->dim + j] = opt->bounds[2 * j];
            } else if (val > opt->bounds[2 * j + 1]) {
                all_positions[i * opt->dim + j] = opt->bounds[2 * j + 1];
            }
        }
    }

    // Sort particles by fitness
    typedef struct { double fitness; int index; } ParticleSort;
    ParticleSort *sort_array = (ParticleSort *)malloc(total_particles * sizeof(ParticleSort));
    if (!sort_array) {
        fprintf(stderr, "🚫 Error: Memory allocation failed for sort_array\n");
        free(sparks);
        free(all_positions);
        free(all_fitness);
        return;
    }

    for (int i = 0; i < total_particles; i++) {
        sort_array[i].fitness = all_fitness[i];
        sort_array[i].index = i;
    }

    // Simple bubble sort (for simplicity; qsort could be used for better performance)
    for (int i = 0; i < total_particles - 1; i++) {
        for (int j = 0; j < total_particles - i - 1; j++) {
            if (sort_array[j].fitness > sort_array[j + 1].fitness) {
                ParticleSort temp = sort_array[j];
                sort_array[j] = sort_array[j + 1];
                sort_array[j + 1] = temp;
            }
        }
    }

    // Select top NUM_PARTICLES
    for (int i = 0; i < opt->population_size; i++) {
        int idx = sort_array[i].index;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = all_positions[idx * opt->dim + j];
        }
        opt->population[i].fitness = all_fitness[idx];
    }

    // Clean up
    free(sparks);
    free(all_positions);
    free(all_fitness);
    free(sort_array);
}

// Main Optimization Function
void FA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->bounds || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs to FA_optimize\n");
        return;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize particles
    initialize_particles(opt);

    // Evaluate initial fitness
    evaluate_fitness(opt, objective_function);

    // Set initial best solution
    int min_idx = 0;
    double min_fitness = opt->population[0].fitness;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < min_fitness) {
            min_fitness = opt->population[i].fitness;
            min_idx = i;
        }
    }
    opt->best_solution.fitness = min_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[min_idx].position[j];
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_particles(opt, objective_function);
        printf("🔄 Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
