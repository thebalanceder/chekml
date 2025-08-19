/* CBO.c - Optimized Implementation file for Colliding Bodies Optimization */
/* Compile with: -O3 -ffast-math for maximum performance */
#include "CBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

/* Collision Phase (Movement towards center of mass) */
void collision_phase(Optimizer *opt) {
    /* Use stack-allocated array for center_of_mass to avoid heap allocation */
    double center_of_mass[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        center_of_mass[j] = 0.0;
    }

    /* Compute center of mass in a single pass */
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            center_of_mass[j] += pos[j];
        }
    }
    double inv_pop_size = 1.0 / opt->population_size;
    for (int j = 0; j < opt->dim; j++) {
        center_of_mass[j] *= inv_pop_size;
    }

    /* Update positions and enforce bounds in a single loop */
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double direction = center_of_mass[j] - pos[j];
            pos[j] += ALPHA * direction;
            /* Inline bound enforcement */
            pos[j] = pos[j] < opt->bounds[2 * j] ? opt->bounds[2 * j] : 
                     pos[j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : pos[j];
        }
    }
}

/* Update Best Solution */
void update_best_solution(Optimizer *opt, double (*objective_function)(double *)) {
    double best_fitness = opt->best_solution.fitness;
    double *best_pos = opt->best_solution.position;

    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        if (new_fitness < best_fitness) {
            best_fitness = new_fitness;
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = pos[j];
            }
        }
    }
    opt->best_solution.fitness = best_fitness;
}

/* Main Optimization Function */
void CBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    /* Seed random number generator once at the start if needed */
    srand((unsigned int)time(NULL));

    for (int iter = 0; iter < opt->max_iter; iter++) {
        collision_phase(opt);
        update_best_solution(opt, objective_function);
    }
}
