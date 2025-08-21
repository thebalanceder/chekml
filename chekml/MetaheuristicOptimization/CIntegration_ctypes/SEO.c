#include "SEO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize population randomly within bounds
void initialize_population_seo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness to be updated later
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all agents
void evaluate_population_seo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Social Engineering Update Phase
void social_engineering_update(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        // Select a random target agent (different from current agent)
        int target_index = rand() % opt->population_size;
        while (target_index == i) {
            target_index = rand() % opt->population_size;
        }

        // Update solution using social engineering formula
        for (int j = 0; j < opt->dim; j++) {
            double randn = rand_double(-1.0, 1.0);  // Approximate normal distribution
            opt->population[i].position[j] += randn * (opt->population[target_index].position[j] - opt->population[i].position[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SEO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Initialize population
    initialize_population_seo(opt);
    evaluate_population_seo(opt, objective_function);

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Perform social engineering update
        social_engineering_update(opt);

        // Evaluate fitness and update population
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness < opt->population[i].fitness) {
                opt->population[i].fitness = new_fitness;
            }

            // Update best solution if better
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
