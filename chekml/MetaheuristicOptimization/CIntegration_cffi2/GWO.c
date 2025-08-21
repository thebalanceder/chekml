#include "GWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Initialize the population of search agents
void initialize_population_gwo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + gwo_rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update Alpha, Beta, and Delta wolves based on fitness
void update_hierarchy(Optimizer *opt, Solution *beta_solution, Solution *delta_solution) {
    double alpha_score = opt->best_solution.fitness;
    double beta_score = beta_solution->fitness;
    double delta_score = delta_solution->fitness;
    int alpha_idx = -1, beta_idx = -1, delta_idx = -1;

    // Find Alpha, Beta, Delta
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

    // Update Alpha (best_solution)
    if (alpha_idx >= 0) {
        opt->best_solution.fitness = alpha_score;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[alpha_idx].position[j];
        }
    }

    // Update Beta
    if (beta_idx >= 0) {
        beta_solution->fitness = beta_score;
        for (int j = 0; j < opt->dim; j++) {
            beta_solution->position[j] = opt->population[beta_idx].position[j];
        }
    }

    // Update Delta
    if (delta_idx >= 0) {
        delta_solution->fitness = delta_score;
        for (int j = 0; j < opt->dim; j++) {
            delta_solution->position[j] = opt->population[delta_idx].position[j];
        }
    }
}

// Update positions of search agents based on Alpha, Beta, and Delta
void gwo_update_positions(Optimizer *opt, double a, Solution *beta_solution, Solution *delta_solution) {
    // Cache frequently accessed pointers
    double *alpha_pos = opt->best_solution.position;
    double *beta_pos = beta_solution->position;
    double *delta_pos = delta_solution->position;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            // Alpha update
            double r1 = gwo_rand_double(0.0, 1.0);
            double r2 = gwo_rand_double(0.0, 1.0);
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            double D_alpha = fabs(C1 * alpha_pos[j] - pos[j]);
            double X1 = alpha_pos[j] - A1 * D_alpha;

            // Beta update
            r1 = gwo_rand_double(0.0, 1.0);
            r2 = gwo_rand_double(0.0, 1.0);
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            double D_beta = fabs(C2 * beta_pos[j] - pos[j]);
            double X2 = beta_pos[j] - A2 * D_beta;

            // Delta update
            r1 = gwo_rand_double(0.0, 1.0);
            r2 = gwo_rand_double(0.0, 1.0);
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            double D_delta = fabs(C3 * delta_pos[j] - pos[j]);
            double X3 = delta_pos[j] - A3 * D_delta;

            // Update position
            pos[j] = (X1 + X2 + X3) / 3.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Allocate Beta and Delta solutions
    Solution beta_solution;
    Solution delta_solution;
    beta_solution.position = (double *)malloc(opt->dim * sizeof(double));
    delta_solution.position = (double *)malloc(opt->dim * sizeof(double));
    beta_solution.fitness = INFINITY;
    delta_solution.fitness = INFINITY;

    // Initialize population
    initialize_population_gwo(opt);

    // Precompute a values for each iteration
    double *a_values = (double *)malloc(opt->max_iter * sizeof(double));
    for (int iter = 0; iter < opt->max_iter; iter++) {
        a_values[iter] = GWO_A_MAX - ((double)iter / opt->max_iter) * (GWO_A_MAX - GWO_A_MIN);
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for all solutions
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }

        // Update hierarchy (Alpha, Beta, Delta)
        update_hierarchy(opt, &beta_solution, &delta_solution);

        // Update positions
        gwo_update_positions(opt, a_values[iter], &beta_solution, &delta_solution);

        // Log progress
        printf("Iteration %d: Best Score = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(a_values);
    free(beta_solution.position);
    free(delta_solution.position);
}
