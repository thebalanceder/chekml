#include "CroSA.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>

// 🐦 Initialize population and memory
void initialize_population_crosa(Optimizer *opt, Solution *memory, double (*objective_function)(double *), XorshiftState_crosa *rng) {
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    // Initialize RNG state
    rng->state = (uint64_t)time(NULL) ^ (uint64_t)opt;

    // Cache best solution
    double best_fitness = INFINITY;
    int best_idx = 0;

    // Initialize population and memory in a single loop
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double *mem_pos = memory[i].position;
        for (int j = 0; j < dim; j++) {
            double lower = bounds[2 * j];
            double upper = bounds[2 * j + 1];
            pos[j] = mem_pos[j] = fast_rand_double(rng, lower, upper);
        }
        double fitness = objective_function(pos);
        opt->population[i].fitness = memory[i].fitness = fitness;

        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
    }

    // Set initial best solution
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    enforce_bound_constraints(opt);
}

// 🐦 Combined update of positions and memory for better cache locality
void update_positions_and_memory(Optimizer *opt, Solution *memory, double (*objective_function)(double *), XorshiftState_crosa *rng) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *bounds = opt->bounds;
    double best_fitness = opt->best_solution.fitness;

    // Precompute constants
    double inv_pop_size = 1.0 / pop_size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        double r = fast_rand_double(rng, 0.0, 1.0);
        int random_crow = (int)(fast_rand_double(rng, 0.0, 1.0) * pop_size);

        if (r > AWARENESS_PROBABILITY) {
            // State 1: Follow another crow
            double *mem_pos = memory[random_crow].position;
            double rand_factor = FLIGHT_LENGTH * fast_rand_double(rng, 0.0, 1.0);
            for (int j = 0; j < dim; j++) {
                pos[j] += rand_factor * (mem_pos[j] - pos[j]);
                // Enforce bounds immediately
                double lower = bounds[2 * j];
                double upper = bounds[2 * j + 1];
                pos[j] = fmax(lower, fmin(upper, pos[j]));
            }
        } else {
            // State 2: Random position within bounds
            for (int j = 0; j < dim; j++) {
                double lower = bounds[2 * j];
                double upper = bounds[2 * j + 1];
                pos[j] = fast_rand_double(rng, lower, upper);
            }
        }

        // Evaluate and update memory
        double new_fitness = objective_function(pos);
        opt->population[i].fitness = new_fitness;

        if (new_fitness < memory[i].fitness) {
            memory[i].fitness = new_fitness;
            for (int j = 0; j < dim; j++) {
                memory[i].position[j] = pos[j];
            }
        }

        // Update best solution
        if (new_fitness < best_fitness) {
            best_fitness = new_fitness;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = pos[j];
            }
            opt->best_solution.fitness = best_fitness;
        }
    }
}

// 🚀 Main Optimization Function
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory array
    Solution *memory = (Solution *)calloc(opt->population_size, sizeof(Solution));
    if (!memory) {
        fprintf(stderr, "Memory allocation failed for memory array\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        memory[i].position = (double *)calloc(opt->dim, sizeof(double));
        if (!memory[i].position) {
            fprintf(stderr, "Memory allocation failed for memory position %d\n", i);
            for (int k = 0; k < i; k++) {
                free(memory[k].position);
            }
            free(memory);
            return;
        }
    }

    // Initialize RNG
    XorshiftState_crosa rng;

    // Initialize population and memory
    initialize_population_crosa(opt, memory, objective_function, &rng);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_positions_and_memory(opt, memory, objective_function, &rng);

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
