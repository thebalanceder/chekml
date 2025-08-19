#include "CroSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 🐦 Optimized random number generation (consider Xorshift for further speedup)
static inline double fast_rand_double(double min, double max) {
    return RAND_DOUBLE(min, max);
}

// 🐦 Initialize population and memory
void initialize_population_crosa(Optimizer *opt, Solution *memory, double (*objective_function)(double *)) {
    // Cache bounds for faster access
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double *mem_pos = memory[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = mem_pos[j] = fast_rand_double(lower, upper);
        }
        double fitness = objective_function(pos);
        opt->population[i].fitness = memory[i].fitness = fitness;
    }

    // Find initial best solution
    int best_idx = 0;
    double best_fitness = opt->population[0].fitness;
    for (int i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, opt->population[best_idx].position, dim * sizeof(double));

    enforce_bound_constraints(opt);
}

// 🐦 Update crow positions
void update_positions(Optimizer *opt, Solution *memory) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;

    // Precompute inverse for random crow selection
    double inv_pop_size = 1.0 / pop_size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double r = fast_rand_double(0.0, 1.0);
        // Use fast integer random for selecting random crow
        int random_crow = (int)(fast_rand_double(0.0, 1.0) * pop_size);

        if (r > AWARENESS_PROBABILITY) {
            // State 1: Follow another crow
            double *mem_pos = memory[random_crow].position;
            double rand_factor = FLIGHT_LENGTH * fast_rand_double(0.0, 1.0);
            for (int j = 0; j < dim; j++) {
                pos[j] += rand_factor * (mem_pos[j] - pos[j]);
            }
        } else {
            // State 2: Random position within bounds
            for (int j = 0; j < dim; j++) {
                double lower = bounds[2 * j];
                double upper = bounds[2 * j + 1];
                pos[j] = fast_rand_double(lower, upper);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// 🐦 Update memory and best solution
void update_memory(Optimizer *opt, Solution *memory, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double best_fitness = opt->best_solution.fitness;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double new_fitness = objective_function(pos);
        opt->population[i].fitness = new_fitness;

        // Update memory if new position is better
        if (new_fitness < memory[i].fitness) {
            memory[i].fitness = new_fitness;
            memcpy(memory[i].position, pos, dim * sizeof(double));
        }

        // Update best solution if necessary
        if (new_fitness < best_fitness) {
            best_fitness = new_fitness;
            memcpy(opt->best_solution.position, pos, dim * sizeof(double));
            opt->best_solution.fitness = best_fitness;
        }
    }
    enforce_bound_constraints(opt);
}

// 🚀 Main Optimization Function
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory array
    Solution *memory = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!memory) {
        fprintf(stderr, "Memory allocation failed for memory array\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        memory[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!memory[i].position) {
            fprintf(stderr, "Memory allocation failed for memory position %d\n", i);
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

        // Log progress (optional, can be disabled for max performance)
        #ifdef LOG_PROGRESS
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
        #endif
    }

    // Cleanup memory
    for (int i = 0; i < opt->population_size; i++) {
        free(memory[i].position);
    }
    free(memory);
}
