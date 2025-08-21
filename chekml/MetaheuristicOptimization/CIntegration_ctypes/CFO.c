/* CFO.c - Implementation file for Central Force Optimization */
#include "CFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Generate random double between min and max
double rand_double(double min, double max);

// Initialize Population
void initialize_population_cfo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                           (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Central Force Update Phase
void central_force_update(Optimizer *opt) {
    double *center_of_mass = (double *)calloc(opt->dim, sizeof(double));
    
    // Calculate center of mass
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            center_of_mass[j] += opt->population[i].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        center_of_mass[j] /= opt->population_size;
    }

    // Update positions
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double direction = center_of_mass[j] - opt->population[i].position[j];
            opt->population[i].position[j] += ALPHA * direction;
        }
    }

    free(center_of_mass);
    enforce_bound_constraints(opt);
}

// Update Best Solution
void update_best_solution_cfo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Main Optimization Function
void CFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_population_cfo(opt);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        central_force_update(opt);
        update_best_solution_cfo(opt, objective_function);
        enforce_bound_constraints(opt);
        
        // Debugging output
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}