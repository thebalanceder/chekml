#include "JOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize subpopulations
void initialize_subpopulations(Optimizer *opt) {
    opt->population_size = NUM_SUBPOPULATIONS * POPULATION_SIZE_PER_SUBPOP;
    opt->population = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!opt->population) {
        fprintf(stderr, "Memory allocation failed for population\n");
        exit(1);
    }

    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!opt->population[i].position) {
            fprintf(stderr, "Memory allocation failed for individual %d\n", i);
            exit(1);
        }

        // Randomly initialize position within bounds
        for (int k = 0; k < opt->dim; k++) {
            opt->population[i].position[k] = rand_double(opt->bounds[2 * k], opt->bounds[2 * k + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }

    // Set optimize function pointer
    opt->optimize = JOA_optimize;
}

// Evaluate fitness for all subpopulations
void evaluate_subpopulations(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;
            double fitness = objective_function(opt->population[idx].position);
            opt->population[idx].fitness = fitness;

            // Update best solution
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                for (int k = 0; k < opt->dim; k++) {
                    opt->best_solution.position[k] = opt->population[idx].position[k];
                }
            }
        }
    }
}

// Update subpopulations
void update_subpopulations(Optimizer *opt) {
    for (int i = 0; i < NUM_SUBPOPULATIONS; i++) {
        int start_idx = i * POPULATION_SIZE_PER_SUBPOP;
        for (int j = 0; j < POPULATION_SIZE_PER_SUBPOP; j++) {
            int idx = start_idx + j;

            // Select a random subpopulation (excluding the current one)
            int other_subpop_idx = rand() % (NUM_SUBPOPULATIONS - 1);
            if (other_subpop_idx >= i) {
                other_subpop_idx++;
            }

            // Select a random individual from the other subpopulation
            int other_start_idx = other_subpop_idx * POPULATION_SIZE_PER_SUBPOP;
            int other_ind_idx = other_start_idx + (rand() % POPULATION_SIZE_PER_SUBPOP);

            // Move towards the selected individual
            for (int k = 0; k < opt->dim; k++) {
                double direction = opt->population[other_ind_idx].position[k] - 
                                  opt->population[idx].position[k];
                opt->population[idx].position[k] += MUTATION_RATE * direction;
            }
        }
    }

    // Enforce bounds
    enforce_bound_constraints(opt);
}

// Main optimization function
void JOA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    srand(time(NULL)); // Seed random number generator

    initialize_subpopulations(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_subpopulations(opt, objective_function);
        update_subpopulations(opt);

        // Debugging output
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up population
    for (int i = 0; i < opt->population_size; i++) {
        free(opt->population[i].position);
    }
    free(opt->population);
    opt->population = NULL;
}
