#include "MSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Function to generate a random number from a normal distribution (Box-Muller transform)
double rand_normal_mso(double mean, double stddev) {
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z;
}

// Update positions (combines exploration and exploitation)
void update_positions_mso(Optimizer *opt, int iter) {
    double p_explore = fmax(MAX_P_EXPLORE * exp(-0.1 * iter), MIN_P_EXPLORE);
    
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double r = rand_double(0.0, 1.0);
            double rand_val = rand_double(0.0, 1.0);
            
            if (rand_val < p_explore) { // Exploration
                if (r < 0.5) {
                    opt->population[i].position[j] = opt->best_solution.position[j] + 
                        rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->best_solution.position[j]);
                } else {
                    opt->population[i].position[j] = opt->best_solution.position[j] - 
                        rand_double(0.0, 1.0) * (opt->best_solution.position[j] - opt->bounds[2 * j]);
                }
            } else { // Exploitation
                double perturbation = rand_normal_mso(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) / PERTURBATION_SCALE;
                opt->population[i].position[j] = opt->best_solution.position[j] + perturbation;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize population (assumed to be done in Optimizer setup)
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
        update_positions_mso(opt, iter);

        // Evaluate fitness and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);
        
        // Optional: Print iteration progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
