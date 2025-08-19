#include "TEO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Fast random double generator (assuming rand_double is not optimized)
double rand_double(double min, double max);

// Initialize the current solution
void initialize_solution(Optimizer *opt, double *solution, double *fitness, double (*objective_function)(double *)) {
    for (int j = 0; j < opt->dim; j++) {
        double min = opt->bounds[2 * j];
        double max = opt->bounds[2 * j + 1];
        solution[j] = min + (max - min) * rand_double(0.0, 1.0);
        // Enforce bounds directly
        if (solution[j] < min) solution[j] = min;
        if (solution[j] > max) solution[j] = max;
    }
    *fitness = objective_function(solution);
    
    // Set best solution
    opt->best_solution.fitness = *fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = solution[j];
    }
}

// Perturb the current solution
void perturb_solution(Optimizer *opt, double *new_solution, double step_size) {
    for (int j = 0; j < opt->dim; j++) {
        // Single uniform perturbation for speed
        double perturbation = step_size * rand_double(-1.0, 1.0);
        new_solution[j] = opt->best_solution.position[j] + perturbation;
        // Enforce bounds directly
        double min = opt->bounds[2 * j];
        double max = opt->bounds[2 * j + 1];
        if (new_solution[j] < min) new_solution[j] = min;
        if (new_solution[j] > max) new_solution[j] = max;
    }
}

// Accept or reject the new solution
void accept_solution(Optimizer *opt, double *new_solution, double new_fitness, double temperature) {
    double delta_fitness = new_fitness - opt->best_solution.fitness;
    if (delta_fitness <= 0 || rand_double(0.0, 1.0) < exp(-delta_fitness / temperature)) {
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = new_solution[j];
        }
        opt->best_solution.fitness = new_fitness;
    }
}

// Main Optimization Function
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL)); // Initialize random seed
    
    // Allocate memory once
    double *current_solution = (double *)malloc(opt->dim * sizeof(double));
    double *new_solution = (double *)malloc(opt->dim * sizeof(double));
    double current_fitness = INFINITY;
    double temperature = TEO_INITIAL_TEMPERATURE;
    double previous_best_fitness = INFINITY;
    int stagnant_iterations = 0;
    const int max_stagnant_iterations = 100; // Early termination criterion

    // Initialize the solution
    initialize_solution(opt, current_solution, &current_fitness, objective_function);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Generate and evaluate new solution
        perturb_solution(opt, new_solution, TEO_STEP_SIZE);
        double new_fitness = objective_function(new_solution);

        // Accept or reject the new solution
        accept_solution(opt, new_solution, new_fitness, temperature);

        // Cool temperature (inlined)
        temperature *= TEO_COOLING_RATE;

        // Early termination: check if fitness hasn't improved significantly
        if (fabs(opt->best_solution.fitness - previous_best_fitness) < 1e-6) {
            stagnant_iterations++;
            if (stagnant_iterations >= max_stagnant_iterations) {
                break;
            }
        } else {
            stagnant_iterations = 0;
            previous_best_fitness = opt->best_solution.fitness;
        }

        // Check for temperature-based convergence
        if (temperature < TEO_FINAL_TEMPERATURE) {
            break;
        }
    }

    // Clean up
    free(current_solution);
    free(new_solution);
}
