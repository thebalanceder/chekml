#include "EPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize penguin positions randomly
void initialize_penguins(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute huddle boundary and temperature profile
void compute_huddle_boundary(double *T_prime, double *R, int t, int max_iter) {
    *T_prime = 2.0 - ((double)t / max_iter);
    *R = rand_double(0.0, 1.0);
}

// Adapt control parameters (f, l, M) based on selected strategy
void adapt_parameters(double *f, double *l, double *M, int strategy, int t, int max_iter) {
    if (strategy == STRATEGY_LINEAR) {
        *f = 2.0 - ((double)t / max_iter) * 1.5;
        *l = 1.5 - ((double)t / max_iter) * 1.0;
        *M = 0.5 + ((double)t / max_iter) * 0.3;
    } else if (strategy == STRATEGY_EXPONENTIAL) {
        *f = 2.0 * exp(-((double)t / (max_iter / 2.0)));
        *l = 1.5 * exp(-((double)t / (max_iter / 3.0)));
        *M = 0.5 * (1.0 + tanh((double)t / (max_iter / 4.0)));
    } else if (strategy == STRATEGY_CHAOTIC) {
        double x = 0.7; // Initial value for logistic map
        for (int i = 0; i < (t % 10); i++) {
            x = 4.0 * x * (1.0 - x);
        }
        *f = 1.5 + x * 0.5;
        *l = 1.0 + x * 0.5;
        *M = 0.3 + x * 0.4;
    }
}

// Update strategy selection probabilities based on historical success
void update_strategy_probabilities(double *strategy_probs, double *strategy_success) {
    double total_success = strategy_success[0] + strategy_success[1] + strategy_success[2] + 1e-10;
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] = strategy_success[i] / total_success;
        if (strategy_probs[i] < 0.1) strategy_probs[i] = 0.1;
        if (strategy_probs[i] > 0.9) strategy_probs[i] = 0.9;
    }
    double sum_probs = strategy_probs[0] + strategy_probs[1] + strategy_probs[2];
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] /= sum_probs;
        strategy_success[i] *= 0.9; // Decay success
    }
}

// Simulate penguin movement in huddle
void huddle_movement(Optimizer *opt, int index, int t, ObjectiveFunction objective_function) {
    double T_prime, R;
    compute_huddle_boundary(&T_prime, &R, t, opt->max_iter);

    // Select adaptation strategy
    static double strategy_probs[STRATEGY_COUNT] = {1.0/3, 1.0/3, 1.0/3};
    static double strategy_success[STRATEGY_COUNT] = {0.0, 0.0, 0.0};
    double r = rand_double(0.0, 1.0);
    int strategy = 0;
    double cumulative = strategy_probs[0];
    for (int i = 0; i < STRATEGY_COUNT - 1; i++) {
        if (r <= cumulative) break;
        strategy++;
        cumulative += strategy_probs[i + 1];
    }

    double f, l, M;
    adapt_parameters(&f, &l, &M, strategy, t, opt->max_iter);

    // Compute distance to best solution
    double D[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        D[j] = fabs(f * rand_double(0.0, 1.0) * opt->best_solution.position[j] - opt->population[index].position[j]);
    }

    // Compute social forces
    double S = M * exp(-((double)t / l)) - exp(-(double)t);

    // Update position
    double new_position[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        new_position[j] = opt->population[index].position[j] + S * D[j] * rand_double(0.0, 1.0);
        if (new_position[j] < opt->bounds[2 * j]) new_position[j] = opt->bounds[2 * j];
        if (new_position[j] > opt->bounds[2 * j + 1]) new_position[j] = opt->bounds[2 * j + 1];
    }

    // Evaluate new solution
    double new_fitness = objective_function(new_position);
    double old_fitness = opt->population[index].fitness;

    // Update position and fitness
    if (new_fitness < old_fitness) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[index].position[j] = new_position[j];
        }
        opt->population[index].fitness = new_fitness;
        strategy_success[strategy] += 1.0;
    }

    // Update strategy probabilities periodically
    if (t % ADAPTATION_INTERVAL == 0 && t > 0) {
        update_strategy_probabilities(strategy_probs, strategy_success);
    }
}

// Main Optimization Function
void EPO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_penguins(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate all penguins
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update penguin positions
        for (int i = 0; i < opt->population_size; i++) {
            huddle_movement(opt, i, iter, objective_function);
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
