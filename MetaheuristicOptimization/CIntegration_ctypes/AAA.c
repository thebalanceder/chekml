#include "AAA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double_aaa(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize Population
void initialize_population_aaa(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double_aaa(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate Population
void evaluate_population_aaa(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Movement Phase
void movement_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double direction = opt->best_solution.position[j] - opt->population[i].position[j];
            opt->population[i].position[j] += STEP_SIZE * direction;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void AAA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_population_aaa(opt);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        evaluate_population_aaa(opt, objective_function);
        
        // Find the best individual
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        
        // Update population positions
        movement_phase(opt);
        
        // Optional: Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
