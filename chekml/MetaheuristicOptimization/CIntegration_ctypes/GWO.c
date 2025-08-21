#include "GWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize the population of search agents
void initialize_population_gwo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY; // Initial fitness
    }
    enforce_bound_constraints(opt);
}

// Update Alpha, Beta, and Delta wolves based on fitness
void update_hierarchy(Optimizer *opt) {
    // Initialize Alpha, Beta, Delta as the first three solutions
    double alpha_score = INFINITY;
    double beta_score = INFINITY;
    double delta_score = INFINITY;
    int alpha_idx = 0, beta_idx = 0, delta_idx = 0;

    // Compute fitness for all solutions
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < alpha_score) {
            delta_score = beta_score;
            delta_idx = beta_idx;
            beta_score = alpha_score;
            beta_idx = alpha_idx;
            alpha_score = fitness;
            alpha_idx = i;
        } else if (fitness < beta_score && fitness > alpha_score) {
            delta_score = beta_score;
            delta_idx = beta_idx;
            beta_score = fitness;
            beta_idx = i;
        } else if (fitness < delta_score && fitness > beta_score) {
            delta_score = fitness;
            delta_idx = i;
        }
    }

    // Update best_solution (Alpha)
    if (alpha_score < opt->best_solution.fitness) {
        opt->best_solution.fitness = alpha_score;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[alpha_idx].position[j];
        }
    }
}

// Update positions of search agents based on Alpha, Beta, and Delta
void gwo_update_positions(Optimizer *opt, int iteration) {
    double a = GWO_A_MAX - ((double)iteration / opt->max_iter) * (GWO_A_MAX - GWO_A_MIN);

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Alpha update
            double r1 = rand_double(0.0, 1.0);
            double r2 = rand_double(0.0, 1.0);
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            double D_alpha = fabs(C1 * opt->best_solution.position[j] - opt->population[i].position[j]);
            double X1 = opt->best_solution.position[j] - A1 * D_alpha;

            // Beta update (using second-best approximation)
            r1 = rand_double(0.0, 1.0);
            r2 = rand_double(0.0, 1.0);
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            double D_beta = fabs(C2 * opt->population[1].position[j] - opt->population[i].position[j]); // Approximation
            double X2 = opt->population[1].position[j] - A2 * D_beta;

            // Delta update (using third-best approximation)
            r1 = rand_double(0.0, 1.0);
            r2 = rand_double(0.0, 1.0);
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            double D_delta = fabs(C3 * opt->population[2].position[j] - opt->population[i].position[j]); // Approximation
            double X3 = opt->population[2].position[j] - A3 * D_delta;

            // Update position
            opt->population[i].position[j] = (X1 + X2 + X3) / 3.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize population
    initialize_population_gwo(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for all solutions
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }

        // Update hierarchy (Alpha, Beta, Delta)
        update_hierarchy(opt);

        // Update positions
        gwo_update_positions(opt, iter);

        // Log progress
        printf("Iteration %d: Best Score = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
