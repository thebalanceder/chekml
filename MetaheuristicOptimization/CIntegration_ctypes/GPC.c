#include "GPC.h"
#include <stdio.h>
#include <string.h>
#include "generaloptimizer.h"

// 🎲 Generate a random number in the given range
double random_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// 📌 Compute stone and worker movement distances
void compute_movement(double velocity, double friction, double* d, double* x) {
    *d = (velocity * velocity) / (2 * G * (sin(THETA) + friction * cos(THETA))); // Stone movement
    *x = (velocity * velocity) / (2 * G * sin(THETA));                          // Worker movement
}

// 📊 Evaluate Fitness of Population
void evaluate_population(Optimizer* opt, ObjectiveFunction objective_function) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// 🔄 Update Population Based on Physics Movement
void update_population(Optimizer* opt) {
    int best_index = 0;

    // Find best solution
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[best_index].fitness) {
            best_index = i;
        }
    }

    // Copy the best solution
    memcpy(opt->best_solution.position, opt->population[best_index].position, opt->dim * sizeof(double));
    opt->best_solution.fitness = opt->population[best_index].fitness;

    // Update other solutions
    for (int i = 0; i < opt->population_size; i++) {
        if (i != best_index) {
            for (int d = 0; d < opt->dim; d++) {
                double velocity = random_uniform(V_MIN, V_MAX);
                double friction = random_uniform(MU_MIN, MU_MAX);
                double d_move, x_move;
                compute_movement(velocity, friction, &d_move, &x_move);
                
                double epsilon = random_uniform(-0.5 * (V_MAX - V_MIN), 0.5 * (V_MAX - V_MIN));
                opt->population[i].position[d] = (opt->population[i].position[d] + d_move) * (x_move * epsilon);
            }
        }
    }
}

// 🚀 Main Optimization Loop
void GPC_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt) return;

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int d = 0; d < opt->dim; d++) {
            double min_bound = opt->bounds[2 * d];
            double max_bound = opt->bounds[2 * d + 1];
            opt->population[i].position[d] = random_uniform(min_bound, max_bound);
        }
        opt->population[i].fitness = INFINITY;
    }

    // Optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_population(opt, objective_function);
        update_population(opt);
        enforce_bound_constraints(opt);
    }
}

