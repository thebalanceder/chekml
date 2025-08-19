#include "WOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize the population of search agents
void initialize_positions(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness
    }
    enforce_bound_constraints(opt);
}

// Update the leader (best solution)
void update_leader(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;  // Update fitness
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Update positions of search agents
void update_positions_woa(Optimizer *opt, int t) {
    double a = WOA_A_INITIAL - t * (WOA_A_INITIAL / (double)opt->max_iter);  // Linearly decreases from 2 to 0
    double a2 = WOA_A2_INITIAL + t * ((WOA_A2_FINAL - WOA_A2_INITIAL) / (double)opt->max_iter);  // Linearly decreases from -1 to -2

    for (int i = 0; i < opt->population_size; i++) {
        double r1 = rand_double(0.0, 1.0);
        double r2 = rand_double(0.0, 1.0);
        double A = 2.0 * a * r1 - a;  // Eq. (2.3)
        double C = 2.0 * r2;          // Eq. (2.4)
        double b = WOA_B;             // Parameter in Eq. (2.5)
        double l = (a2 - 1.0) * rand_double(0.0, 1.0) + 1.0;  // Parameter in Eq. (2.5)
        double p = rand_double(0.0, 1.0);  // Strategy selection

        for (int j = 0; j < opt->dim; j++) {
            if (p < 0.5) {
                if (fabs(A) >= 1.0) {  // Search for prey (exploration)
                    int rand_leader_index = (int)(rand_double(0.0, 1.0) * opt->population_size);
                    double X_rand_j = opt->population[rand_leader_index].position[j];
                    double D_X_rand = fabs(C * X_rand_j - opt->population[i].position[j]);  // Eq. (2.7)
                    opt->population[i].position[j] = X_rand_j - A * D_X_rand;  // Eq. (2.8)
                } else {  // Encircling prey (exploitation)
                    double D_Leader = fabs(C * opt->best_solution.position[j] - opt->population[i].position[j]);  // Eq. (2.1)
                    opt->population[i].position[j] = opt->best_solution.position[j] - A * D_Leader;  // Eq. (2.2)
                }
            } else {  // Spiral bubble-net attack
                double distance2Leader = fabs(opt->best_solution.position[j] - opt->population[i].position[j]);
                opt->population[i].position[j] = distance2Leader * exp(b * l) * cos(l * 2.0 * M_PI) + opt->best_solution.position[j];  // Eq. (2.5)
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Initialize positions
    initialize_positions(opt);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Update leader and fitness
        update_leader(opt, objective_function);

        // Update positions
        update_positions_woa(opt, t);

        // Log progress
        printf("Iteration %d: Best Score = %f\n", t + 1, opt->best_solution.fitness);
    }
}
