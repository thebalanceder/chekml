#include "TLBO.h"
#include "generaloptimizer.h"
#include <stdint.h>

// Custom Linear Congruential Generator (LCG) for fast random numbers
static uint64_t lcg_state = 1;
static inline double fast_rand(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 32) / 0x100000000ULL;
}

static inline double rand_double(double min, double max) {
    return min + (max - min) * fast_rand();
}

// Main Optimization Function
void TLBO_optimize(Optimizer *restrict opt, ObjectiveFunction objective_function) {
    int i, j, iter;
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    double *restrict bounds = opt->bounds;

    // Preallocate buffers
    double *restrict mean_student = (double *)calloc(dim, sizeof(double));
    double *restrict new_solution = (double *)malloc(dim * sizeof(double));
    int *restrict partners = (int *)malloc(pop_size * sizeof(int));
    double *restrict rand_buffer = (double *)malloc(pop_size * dim * 2 * sizeof(double)); // Precompute random values
    int best_idx = 0;

    // Initialize population fitness and best solution
    for (i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            best_idx = i;
            for (j = 0; j < dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
        partners[i] = i;
    }

    // Main optimization loop
    for (iter = 0; iter < max_iter; iter++) {
        // Precompute random values for this iteration
        for (i = 0; i < pop_size * dim * 2; i++) {
            rand_buffer[i] = fast_rand();
        }
        int rand_idx = 0;

        // Shuffle partners
        for (i = pop_size - 1; i > 0; i--) {
            int j = (int)(rand_buffer[rand_idx++] * (i + 1));
            int temp = partners[i];
            partners[i] = partners[j];
            partners[j] = temp;
        }

        // Compute mean of the population
        for (j = 0; j < dim; j++) {
            mean_student[j] = 0.0;
        }
        for (i = 0; i < pop_size; i++) {
            double *restrict pos = opt->population[i].position;
            for (j = 0; j < dim; j++) {
                mean_student[j] += pos[j] / pop_size;
            }
        }

        // Combined teacher and learner phases
        for (i = 0; i < pop_size; i++) {
            double *restrict current_pos = opt->population[i].position;
            double *restrict teacher_pos = opt->population[best_idx].position;
            double *restrict partner_pos = opt->population[partners[i]].position;
            double new_fitness;
            int tf = 1 + (int)(rand_buffer[rand_idx++]); // Teaching factor (1 or 2)

            // Teacher phase
            for (j = 0; j < dim; j++) {
                new_solution[j] = current_pos[j] + rand_buffer[rand_idx++] * (teacher_pos[j] - tf * mean_student[j]);
                new_solution[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], new_solution[j]));
            }
            new_fitness = objective_function(new_solution);
            if (new_fitness < opt->population[i].fitness) {
                for (j = 0; j < dim; j++) {
                    current_pos[j] = new_solution[j];
                }
                opt->population[i].fitness = new_fitness;
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    best_idx = i;
                    for (j = 0; j < dim; j++) {
                        opt->best_solution.position[j] = current_pos[j];
                    }
                }
            }

            // Learner phase
            if (opt->population[i].fitness < opt->population[partners[i]].fitness) {
                for (j = 0; j < dim; j++) {
                    new_solution[j] = current_pos[j] + rand_buffer[rand_idx++] * (current_pos[j] - partner_pos[j]);
                    new_solution[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], new_solution[j]));
                }
            } else {
                for (j = 0; j < dim; j++) {
                    new_solution[j] = current_pos[j] + rand_buffer[rand_idx++] * (partner_pos[j] - current_pos[j]);
                    new_solution[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], new_solution[j]));
                }
            }
            new_fitness = objective_function(new_solution);
            if (new_fitness < opt->population[i].fitness) {
                for (j = 0; j < dim; j++) {
                    current_pos[j] = new_solution[j];
                }
                opt->population[i].fitness = new_fitness;
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    best_idx = i;
                    for (j = 0; j < dim; j++) {
                        opt->best_solution.position[j] = current_pos[j];
                    }
                }
            }
        }
    }

    // Cleanup
    free(rand_buffer);
    free(partners);
    free(new_solution);
    free(mean_student);
}
