#include "SCA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Initialize Solutions
void initialize_solutions(Optimizer *opt) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * ((double)rand() / RAND_MAX);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update Position using Sine-Cosine Equations
void sca_update_position(Optimizer *opt, int t, double r1_factor) {
    const double two_pi = 6.283185307179586; // Precompute 2 * pi

    // Cache best solution position for faster access
    double *best_pos = opt->best_solution.position;

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double r2 = two_pi * ((double)rand() / RAND_MAX);
            double r3 = 2.0 * ((double)rand() / RAND_MAX);
            double r4 = (double)rand() / RAND_MAX;

            double delta = r3 * best_pos[j] - pos[j];
            pos[j] += r1_factor * (r4 < 0.5 ? sin(r2) : cos(r2)) * fabs(delta);
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("SCA is optimizing your problem...\n");

    // Initialize solutions
    initialize_solutions(opt);

    // Evaluate initial solutions and find best
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }

    // Precompute constant for r1 calculation
    const double r1_decrement = SCA_A / opt->max_iter;

    // Main loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Compute r1 once per iteration
        double r1_factor = SCA_A - t * r1_decrement;

        // Update positions
        sca_update_position(opt, t, r1_factor);

        // Evaluate solutions and update best
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
