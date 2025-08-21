#include "SCA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Solutions
void initialize_solutions(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update Position using Sine-Cosine Equations
void sca_update_position(Optimizer *opt, int t) {
    double r1 = SCA_A - t * (SCA_A / opt->max_iter); // Linearly decreasing r1
    double pi = 3.141592653589793;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double r2 = 2 * pi * rand_double(0.0, 1.0);
            double r3 = 2 * rand_double(0.0, 1.0);
            double r4 = rand_double(0.0, 1.0);

            if (r4 < 0.5) {
                // Sine update
                opt->population[i].position[j] += r1 * sin(r2) * 
                    fabs(r3 * opt->best_solution.position[j] - opt->population[i].position[j]);
            } else {
                // Cosine update
                opt->population[i].position[j] += r1 * cos(r2) * 
                    fabs(r3 * opt->best_solution.position[j] - opt->population[i].position[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("SCA is optimizing your problem...\n");

    // Initialize solutions
    initialize_solutions(opt);

    // Evaluate initial solutions
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Update positions
        sca_update_position(opt, t);

        // Evaluate new solutions
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

        // Display progress every 50 iterations
        if ((t + 1) % 50 == 0) {
            printf("At iteration %d, the optimum is %f\n", t + 1, opt->best_solution.fitness);
        }
    }

    printf("The best solution obtained by SCA is: [");
    for (int j = 0; j < opt->dim - 1; j++) {
        printf("%f, ", opt->best_solution.position[j]);
    }
    printf("%f]\n", opt->best_solution.position[opt->dim - 1]);
    printf("The best optimal value of the objective function found by SCA is: %f\n", 
           opt->best_solution.fitness);
}
