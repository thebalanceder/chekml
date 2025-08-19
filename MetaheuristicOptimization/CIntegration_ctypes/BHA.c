#include "BHA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <math.h>    // For sqrt()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize star positions randomly within bounds
void initialize_stars(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY;  // Will be updated by objective function
    }
    enforce_bound_constraints(opt);
}

// Update star positions based on black hole attraction
void update_star_positions(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        // Skip the black hole itself (best solution)
        if (opt->population[i].fitness == opt->best_solution.fitness) {
            continue;
        }
        double landa = rand_double(0.0, 1.0);
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += rand_double(0.0, 1.0) * 
                                             (opt->best_solution.position[j] - opt->population[i].position[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Replace black hole if a star has better fitness
void replace_with_better_black_hole(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Replace stars that cross the event horizon with new random stars
void new_star_generation(Optimizer *opt) {
    double total_fitness = 0.0;
    for (int i = 0; i < opt->population_size; i++) {
        total_fitness += opt->population[i].fitness;
    }
    double R = opt->best_solution.fitness / total_fitness;  // Event horizon radius

    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness == opt->best_solution.fitness) {
            continue;  // Skip the black hole
        }
        double dist = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->best_solution.position[j] - opt->population[i].position[j];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < R) {
            for (int j = 0; j < opt->dim; j++) {
                double lb = opt->bounds[2 * j];
                double ub = opt->bounds[2 * j + 1];
                opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
            }
            opt->population[i].fitness = INFINITY;  // Will be updated by objective function
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void BHA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize stars
    initialize_stars(opt);
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
        update_star_positions(opt);
        replace_with_better_black_hole(opt);
        new_star_generation(opt);

        // Evaluate fitness for all stars
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
        enforce_bound_constraints(opt);

        // Optional: Print progress
        printf("Iteration %d: Best Fitness = %g\n", iter + 1, opt->best_solution.fitness);
    }
}
