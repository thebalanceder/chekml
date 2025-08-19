#include "TLBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Teacher Phase
void teacher_phase(Optimizer *opt, ObjectiveFunction objective_function) {
    int i, j;
    double *mean_student = (double *)calloc(opt->dim, sizeof(double));
    int best_idx = 0;
    double best_fitness = opt->population[0].fitness;

    // Find the teacher (best solution)
    for (i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    // Compute mean of the population
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            mean_student[j] += opt->population[i].position[j] / opt->population_size;
        }
    }

    // Apply teacher phase for each student
    for (i = 0; i < opt->population_size; i++) {
        int tf = 1 + (rand() % 2); // Teaching factor (1 or 2)
        double *new_solution = (double *)malloc(opt->dim * sizeof(double));
        double new_fitness;

        // Generate new solution
        for (j = 0; j < opt->dim; j++) {
            new_solution[j] = opt->population[i].position[j] +
                              rand_double(0.0, 1.0) * (opt->population[best_idx].position[j] - tf * mean_student[j]);
            // Bound the solution
            if (new_solution[j] < opt->bounds[2 * j]) {
                new_solution[j] = opt->bounds[2 * j];
            } else if (new_solution[j] > opt->bounds[2 * j + 1]) {
                new_solution[j] = opt->bounds[2 * j + 1];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(new_solution);

        // Greedy selection
        if (new_fitness < opt->population[i].fitness) {
            for (j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_solution[j];
            }
            opt->population[i].fitness = new_fitness;
        }

        free(new_solution);
    }

    free(mean_student);
}

// Learner Phase
void learner_phase(Optimizer *opt, ObjectiveFunction objective_function) {
    int i, j;
    // Generate random permutation for partners
    int *partners = (int *)malloc(opt->population_size * sizeof(int));
    for (i = 0; i < opt->population_size; i++) {
        partners[i] = i;
    }
    // Shuffle partners
    for (i = opt->population_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = partners[i];
        partners[i] = partners[j];
        partners[j] = temp;
    }

    // Apply learner phase for each student
    for (i = 0; i < opt->population_size; i++) {
        int partner_idx = partners[i];
        double *new_solution = (double *)malloc(opt->dim * sizeof(double));
        double new_fitness;

        // Generate new solution
        if (opt->population[i].fitness < opt->population[partner_idx].fitness) {
            for (j = 0; j < opt->dim; j++) {
                new_solution[j] = opt->population[i].position[j] +
                                  rand_double(0.0, 1.0) * (opt->population[i].position[j] - opt->population[partner_idx].position[j]);
            }
        } else {
            for (j = 0; j < opt->dim; j++) {
                new_solution[j] = opt->population[i].position[j] +
                                  rand_double(0.0, 1.0) * (opt->population[partner_idx].position[j] - opt->population[i].position[j]);
            }
        }

        // Bound the solution
        for (j = 0; j < opt->dim; j++) {
            if (new_solution[j] < opt->bounds[2 * j]) {
                new_solution[j] = opt->bounds[2 * j];
            } else if (new_solution[j] > opt->bounds[2 * j + 1]) {
                new_solution[j] = opt->bounds[2 * j + 1];
            }
        }

        // Evaluate new solution
        new_fitness = objective_function(new_solution);

        // Greedy selection
        if (new_fitness < opt->population[i].fitness) {
            for (j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = new_solution[j];
            }
            opt->population[i].fitness = new_fitness;
        }

        free(new_solution);
    }

    free(partners);
}

// Main Optimization Function
void TLBO_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
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
        teacher_phase(opt, objective_function);
        learner_phase(opt, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Enforce bounds
        enforce_bound_constraints(opt);
    }
}
