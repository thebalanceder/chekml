#include "TEO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize the current solution (randomly within bounds)
void initialize_solution(Optimizer *opt, double *current_solution, double *current_fitness, double (*objective_function)(double *)) {
    for (int j = 0; j < opt->dim; j++) {
        current_solution[j] = opt->bounds[2 * j] + (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * rand_double(0.0, 1.0);
    }
    *current_fitness = objective_function(current_solution);
    enforce_bound_constraints(opt);
    
    // Set best solution initially
    opt->best_solution.fitness = *current_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = current_solution[j];
    }
}

// Perturb the current solution
void perturb_solution(Optimizer *opt, double *new_solution) {
    for (int j = 0; j < opt->dim; j++) {
        // Gaussian perturbation (approximated using uniform random numbers)
        double perturbation = STEP_SIZE * (rand_double(-1.0, 1.0) + rand_double(-1.0, 1.0)) / 2.0;
        new_solution[j] = opt->best_solution.position[j] + perturbation;
    }
    enforce_bound_constraints(opt);
}

// Accept or reject the new solution based on fitness and temperature
void accept_solution(Optimizer *opt, double *new_solution, double new_fitness, double temperature) {
    double delta_fitness = new_fitness - opt->best_solution.fitness;
    if (delta_fitness < 0 || rand_double(0.0, 1.0) < exp(-delta_fitness / temperature)) {
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = new_solution[j];
        }
        opt->best_solution.fitness = new_fitness;
    }
}

// Update the best solution (if current is better)
void update_best_solution_teo(Optimizer *opt) {
    // Already handled in accept_solution for TEO
}

// Cool the temperature
void cool_temperature(double *temperature) {
    *temperature *= COOLING_RATE;
}

// Main Optimization Function
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *current_solution = (double *)malloc(opt->dim * sizeof(double));
    double current_fitness = INFINITY;
    double temperature = INITIAL_TEMPERATURE;

    // Initialize the solution
    initialize_solution(opt, current_solution, &current_fitness, objective_function);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Generate and evaluate new solution
        double *new_solution = (double *)malloc(opt->dim * sizeof(double));
        perturb_solution(opt, new_solution);
        double new_fitness = objective_function(new_solution);

        // Accept or reject the new solution
        accept_solution(opt, new_solution, new_fitness, temperature);

        // Cool the temperature
        cool_temperature(&temperature);

        // Check for convergence
        if (temperature < FINAL_TEMPERATURE) {
            free(new_solution);
            break;
        }

        free(new_solution);
    }

    free(current_solution);
}
