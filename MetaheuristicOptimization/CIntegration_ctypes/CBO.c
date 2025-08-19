/* CBO.c - Implementation file for Colliding Bodies Optimization */
#include "CBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Collision Phase (Movement towards center of mass)
void collision_phase(Optimizer *opt) {
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
void update_best_solution(Optimizer *opt, double (*objective_function)(double *)) {
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
void CBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    for (int iter = 0; iter < opt->max_iter; iter++) {
        collision_phase(opt);
        update_best_solution(opt, objective_function);
        enforce_bound_constraints(opt);
    }
}