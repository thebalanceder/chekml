#include "TLBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Inline function to generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Teacher Phase
void teacher_phase(Optimizer *restrict opt, ObjectiveFunction objective_function, int *restrict partners) {
    int i, j;
    double *restrict mean_student = (double *)calloc(opt->dim, sizeof(double));
    double *restrict new_solution = (double *)malloc(opt->dim * sizeof(double));
    int best_idx = 0;
    double best_fitness = opt->population[0].fitness;

    // Find the teacher (best solution)
    for (i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    // Compute mean of the population
    for (i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (j = 0; j < opt->dim; j++) {
            mean_student[j] += pos[j] / opt->population_size;
        }
    }

    // Apply teacher phase for each student
    for (i = 0; i < opt->population_size; i++) {
        int tf = 1 + (rand() & 1); // Faster teaching factor (1 or 2)
        double *restrict current_pos = opt->population[i].position;
        double *restrict teacher_pos = opt->population[best_idx].position;
        double new_fitness;

        // Generate new solution with inline bounds checking
        for (j = 0; j < opt->dim; j++) {
            new_solution[j] = current_pos[j] + rand_double(0.0, 1.0) * (teacher_pos[j] - tf * mean_student[j]);
            new_solution[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_solution[j]));
        }

        // Evaluate and update if better
        new_fitness = objective_function(new_solution);
        if (new_fitness < opt->population[i].fitness) {
            for (j = 0; j < opt->dim; j++) {
                current_pos[j] = new_solution[j];
            }
            opt->population[i].fitness = new_fitness;
            partners[i] = best_idx; // Update partner to reflect potential new best
        }
    }

    free(new_solution);
    free(mean_student);
}

// Learner Phase
void learner_phase(Optimizer *restrict opt, ObjectiveFunction objective_function, int *restrict partners) {
    int i, j;
    double *restrict new_solution = (double *)malloc(opt->dim * sizeof(double));

    // Apply learner phase for each student
    for (i = 0; i < opt->population_size; i++) {
        int partner_idx = partners[i];
        double *restrict current_pos = opt->population[i].position;
        double *restrict partner_pos = opt->population[partner_idx].position;
        double new_fitness;

        // Generate new solution
        if (opt->population[i].fitness < opt->population[partner_idx].fitness) {
            for (j = 0; j < opt->dim; j++) {
                new_solution[j] = current_pos[j] + rand_double(0.0, 1.0) * (current_pos[j] - partner_pos[j]);
                new_solution[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_solution[j]));
            }
        } else {
            for (j = 0; j < opt->dim; j++) {
                new_solution[j] = current_pos[j] + rand_double(0.0, 1.0) * (partner_pos[j] - current_pos[j]);
                new_solution[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_solution[j]));
            }
        }

        // Evaluate and update if better
        new_fitness = objective_function(new_solution);
        if (new_fitness < opt->population[i].fitness) {
            for (j = 0; j < opt->dim; j++) {
                current_pos[j] = new_solution[j];
            }
            opt->population[i].fitness = new_fitness;
        }
    }

    free(new_solution);
}

// Main Optimization Function
void TLBO_optimize(Optimizer *restrict opt, ObjectiveFunction objective_function) {
    int i, j, iter;
    int *restrict partners = (int *)malloc(opt->population_size * sizeof(int));

    // Initialize population fitness and best solution
    for (i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
        partners[i] = i;
    }

    // Main optimization loop
    for (iter = 0; iter < opt->max_iter; iter++) {
        // Shuffle partners efficiently
        for (i = opt->population_size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = partners[i];
            partners[i] = partners[j];
            partners[j] = temp;
        }

        teacher_phase(opt, objective_function, partners);
        learner_phase(opt, objective_function, partners);

        // Update best solution
        for (i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }

    free(partners);
}
