#include "EVO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Inline random double generator
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Quicksort implementation
static void quicksort_indices(double *fitness, int *indices, int low, int high) {
    if (low < high) {
        double pivot = fitness[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (fitness[indices[j]] <= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        int pi = i + 1;
        quicksort_indices(fitness, indices, low, pi - 1);
        quicksort_indices(fitness, indices, pi + 1, high);
    }
}

// Initialize particle positions and velocities
void evo_initialize_particles(Optimizer *opt, EVO_Particle *particles) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            particles[i].velocity[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            particles[i].gradient[j] = 0.0;
        }
        particles[i].position = opt->population[i].position;
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all particles
void evaluate_fitness_evo(Optimizer *opt, double (*objective_function)(double *)) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Compute gradient using finite differences
void compute_gradient(Optimizer *opt, EVO_Particle *particles, double (*objective_function)(double *)) {
    double x_plus[opt->dim], x_minus[opt->dim];
    
    for (int i = 0; i < opt->population_size; i++) {
        double* pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            x_plus[j] = pos[j];
            x_minus[j] = pos[j];
        }
        for (int j = 0; j < opt->dim; j++) {
            x_plus[j] += LEARNING_RATE;
            x_minus[j] -= LEARNING_RATE;
            particles[i].gradient[j] = (objective_function(x_plus) - objective_function(x_minus)) / (2 * LEARNING_RATE);
            x_plus[j] = pos[j];
            x_minus[j] = pos[j];
        }
    }
}

// Update velocities and positions
void update_velocity_and_position(Optimizer *opt, EVO_Particle *particles) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            particles[i].velocity[j] = MOMENTUM * particles[i].velocity[j] + 
                                      STEP_SIZE * particles[i].gradient[j];
            opt->population[i].position[j] -= particles[i].velocity[j];
        }
    }
    enforce_bound_constraints(opt);
}

// Free EVO particle arrays
void free_evo_particles(EVO_Particle *particles, int population_size) {
    for (int i = 0; i < population_size; i++) {
        free(particles[i].velocity);
        free(particles[i].gradient);
        particles[i].position = NULL;
    }
    free(particles);
}

// Main Optimization Function
void EVO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL)); // Seed random number generator once

    // Allocate EVO-specific particle data
    EVO_Particle *particles = (EVO_Particle *)malloc(opt->population_size * sizeof(EVO_Particle));
    if (!particles) {
        fprintf(stderr, "EVO_optimize: Memory allocation failed for particles\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        particles[i].velocity = (double *)malloc(opt->dim * sizeof(double));
        particles[i].gradient = (double *)malloc(opt->dim * sizeof(double));
        if (!particles[i].velocity || !particles[i].gradient) {
            fprintf(stderr, "EVO_optimize: Memory allocation failed for particle %d\n", i);
            free_evo_particles(particles, i + 1);
            return;
        }
    }

    // Initialize reusable arrays
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sorted_indices = (int *)malloc(opt->population_size * sizeof(int));
    EVO_Particle *temp_particles = (EVO_Particle *)malloc(opt->population_size * sizeof(EVO_Particle));
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    if (!fitness || !sorted_indices || !temp_particles || !temp_fitness) {
        fprintf(stderr, "EVO_optimize: Memory allocation failed for arrays\n");
        free_evo_particles(particles, opt->population_size);
        free(fitness);
        free(sorted_indices);
        free(temp_particles);
        free(temp_fitness);
        return;
    }

    // Initialize particles
    evo_initialize_particles(opt, particles);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        evaluate_fitness_evo(opt, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Store fitness and indices
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = opt->population[i].fitness;
            sorted_indices[i] = i;
        }

        // Sort indices by fitness using quicksort
        quicksort_indices(fitness, sorted_indices, 0, opt->population_size - 1);

        // Reorder particles and update fitness
        for (int i = 0; i < opt->population_size; i++) {
            temp_particles[i] = particles[sorted_indices[i]];
            temp_fitness[i] = fitness[sorted_indices[i]];
        }
        for (int i = 0; i < opt->population_size; i++) {
            particles[i] = temp_particles[i];
            opt->population[i].fitness = temp_fitness[i];
            particles[i].position = opt->population[i].position;
        }

        // Compute gradient
        compute_gradient(opt, particles, objective_function);

        // Update velocities and positions
        update_velocity_and_position(opt, particles);

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(fitness);
    free(sorted_indices);
    free(temp_particles);
    free(temp_fitness);
    free_evo_particles(particles, opt->population_size);
}
