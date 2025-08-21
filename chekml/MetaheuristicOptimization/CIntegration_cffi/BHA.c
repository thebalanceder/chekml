#include "BHA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <math.h>    // For sqrt()

// Inline function to generate a random double between min and max
inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize star positions randomly within bounds and evaluate fitness
void initialize_stars(Optimizer *opt, double (*objective_function)(double *)) {
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double best_fitness = INFINITY;
    int best_idx = 0;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        double fitness = objective_function(pos);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }
    enforce_bound_constraints(opt);
}

// Update star positions based on black hole attraction
void update_star_positions(Optimizer *opt, int black_hole_idx) {
    double *best_pos = opt->best_solution.position;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        if (i == black_hole_idx) {
            continue;  // Skip the black hole
        }
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] += rand_double(0.0, 1.0) * (best_pos[j] - pos[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Replace black hole if a star has better fitness and update fitness values
void replace_with_better_black_hole(Optimizer *opt, int *black_hole_idx) {
    double best_fitness = opt->best_solution.fitness;
    int new_best_idx = *black_hole_idx;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            new_best_idx = i;
        }
    }
    if (new_best_idx != *black_hole_idx) {
        opt->best_solution.fitness = best_fitness;
        for (int j = 0; j < dim; j++) {
            opt->best_solution.position[j] = opt->population[new_best_idx].position[j];
        }
        *black_hole_idx = new_best_idx;
    }
}

// Replace stars that cross the event horizon with new random stars
void new_star_generation(Optimizer *opt, int black_hole_idx, double (*objective_function)(double *)) {
    double total_fitness = 0.0;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *bounds = opt->bounds;

    // Compute total fitness
    for (int i = 0; i < pop_size; i++) {
        total_fitness += opt->population[i].fitness;
    }
    double R = opt->best_solution.fitness / total_fitness;  // Event horizon radius
    double R_squared = R * R;  // Use squared distance to avoid sqrt

    double *best_pos = opt->best_solution.position;
    for (int i = 0; i < pop_size; i++) {
        if (i == black_hole_idx) {
            continue;  // Skip the black hole
        }
        double *pos = opt->population[i].position;
        double dist_squared = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = best_pos[j] - pos[j];
            dist_squared += diff * diff;
        }
        if (dist_squared < R_squared) {
            for (int j = 0; j < dim; j++) {
                double lb = bounds[2 * j];
                double ub = bounds[2 * j + 1];
                pos[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
            }
            opt->population[i].fitness = objective_function(pos);
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void BHA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    static int initialized = 0;
    if (!initialized) {
        srand(time(NULL));  // Seed random number generator once
        initialized = 1;
    }

    int black_hole_idx = 0;
    initialize_stars(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_star_positions(opt, black_hole_idx);
        
        // Update fitness and check for new black hole
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        replace_with_better_black_hole(opt, &black_hole_idx);
        new_star_generation(opt, black_hole_idx, objective_function);

        // Optional: Print progress
        printf("Iteration %d: Best Fitness = %g\n", iter + 1, opt->best_solution.fitness);
    }
}
