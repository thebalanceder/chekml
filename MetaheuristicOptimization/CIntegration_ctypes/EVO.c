#include "EVO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize particle positions and velocities
void evo_initialize_particles(Optimizer *opt, EVO_Particle *particles) {
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
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Compute gradient using finite differences
void compute_gradient(Optimizer *opt, EVO_Particle *particles, double (*objective_function)(double *)) {
    double *x_plus = (double *)malloc(opt->dim * sizeof(double));
    double *x_minus = (double *)malloc(opt->dim * sizeof(double));

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Copy current position
            for (int k = 0; k < opt->dim; k++) {
                x_plus[k] = opt->population[i].position[k];
                x_minus[k] = opt->population[i].position[k];
            }
            // Perturb dimension j
            x_plus[j] += LEARNING_RATE;
            x_minus[j] -= LEARNING_RATE;
            // Compute gradient
            particles[i].gradient[j] = (objective_function(x_plus) - objective_function(x_minus)) / (2 * LEARNING_RATE);
        }
    }

    free(x_plus);
    free(x_minus);
}

// Update velocities and positions
void update_velocity_and_position(Optimizer *opt, EVO_Particle *particles) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Update velocity: momentum * velocity + step_size * gradient
            particles[i].velocity[j] = MOMENTUM * particles[i].velocity[j] + 
                                      STEP_SIZE * particles[i].gradient[j];
            // Update position: position - velocity
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
        particles[i].position = NULL; // Prevent dangling pointer
    }
    free(particles);
}

// Main Optimization Function
void EVO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
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

    // Initialize particles
    evo_initialize_particles(opt, particles);

    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sorted_indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!fitness || !sorted_indices) {
        fprintf(stderr, "EVO_optimize: Memory allocation failed for fitness or sorted_indices\n");
        free_evo_particles(particles, opt->population_size);
        free(fitness);
        free(sorted_indices);
        return;
    }

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

        // Store fitness and indices for sorting
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = opt->population[i].fitness;
            sorted_indices[i] = i;
        }

        // Bubble sort indices by fitness
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (fitness[sorted_indices[j]] > fitness[sorted_indices[j + 1]]) {
                    int temp = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[j + 1];
                    sorted_indices[j + 1] = temp;
                }
            }
        }

        // Reorder particles and update fitness without changing position pointers
        EVO_Particle *temp_particles = (EVO_Particle *)malloc(opt->population_size * sizeof(EVO_Particle));
        if (!temp_particles) {
            fprintf(stderr, "EVO_optimize: Memory allocation failed for temp_particles\n");
            free_evo_particles(particles, opt->population_size);
            free(fitness);
            free(sorted_indices);
            return;
        }
        double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
        if (!temp_fitness) {
            fprintf(stderr, "EVO_optimize: Memory allocation failed for temp_fitness\n");
            free(temp_particles);
            free_evo_particles(particles, opt->population_size);
            free(fitness);
            free(sorted_indices);
            return;
        }

        // Copy to temporary arrays
        for (int i = 0; i < opt->population_size; i++) {
            temp_particles[i] = particles[sorted_indices[i]];
            temp_fitness[i] = fitness[sorted_indices[i]];
        }

        // Update particles and fitness
        for (int i = 0; i < opt->population_size; i++) {
            particles[i] = temp_particles[i];
            opt->population[i].fitness = temp_fitness[i];
            particles[i].position = opt->population[i].position; // Maintain position reference
        }

        free(temp_particles);
        free(temp_fitness);

        // Recompute fitness to ensure consistency with positions
        evaluate_fitness_evo(opt, objective_function);

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
    free_evo_particles(particles, opt->population_size);
}
