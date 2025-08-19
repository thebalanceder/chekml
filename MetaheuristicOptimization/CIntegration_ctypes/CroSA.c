#include "CroSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize population and memory
void initialize_population_crosa(Optimizer *opt, Solution *memory, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lower_bound = opt->bounds[2 * j];
            double upper_bound = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lower_bound + (upper_bound - lower_bound) * rand_double(0.0, 1.0);
            memory[i].position[j] = opt->population[i].position[j]; // Initialize memory
        }
        // Evaluate initial fitness
        opt->population[i].fitness = objective_function(opt->population[i].position);
        memory[i].fitness = opt->population[i].fitness;
    }
    // Initialize best solution
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }
    enforce_bound_constraints(opt);
}

// Update crow positions
void update_positions(Optimizer *opt, Solution *memory) {
    for (int i = 0; i < opt->population_size; i++) {
        double r = rand_double(0.0, 1.0);
        int random_crow = rand() % opt->population_size; // Random crow to follow

        if (r > AWARENESS_PROBABILITY) {
            // State 1: Follow another crow
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] += FLIGHT_LENGTH * rand_double(0.0, 1.0) * 
                                                 (memory[random_crow].position[j] - opt->population[i].position[j]);
            }
        } else {
            // State 2: Random position within bounds
            for (int j = 0; j < opt->dim; j++) {
                double lower_bound = opt->bounds[2 * j];
                double upper_bound = opt->bounds[2 * j + 1];
                opt->population[i].position[j] = lower_bound + (upper_bound - lower_bound) * rand_double(0.0, 1.0);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update memory and best solution
void update_memory(Optimizer *opt, Solution *memory, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = new_fitness;

        // Update memory if new position is better
        if (new_fitness < memory[i].fitness) {
            memory[i].fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                memory[i].position[j] = opt->population[i].position[j];
            }
        }

        // Update best solution if necessary
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory for the memory array
    Solution *memory = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!memory) {
        fprintf(stderr, "Memory allocation failed for memory array\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        memory[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!memory[i].position) {
            fprintf(stderr, "Memory allocation failed for memory position %d\n", i);
            // Cleanup
            for (int k = 0; k < i; k++) {
                free(memory[k].position);
            }
            free(memory);
            return;
        }
    }

    // Initialize population and memory
    initialize_population_crosa(opt, memory, objective_function);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_positions(opt, memory);
        update_memory(opt, memory, objective_function);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Cleanup memory
    for (int i = 0; i < opt->population_size; i++) {
        free(memory[i].position);
    }
    free(memory);
}
