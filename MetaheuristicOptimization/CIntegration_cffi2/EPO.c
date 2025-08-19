#include "EPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize penguin positions randomly
void initialize_penguins(Optimizer *opt) {
    #pragma omp parallel for
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

// Adapt control parameters (f, l, m_param) based on selected strategy
void adapt_parameters(double *f, double *l, double *m_param, int strategy, double t_norm, int t) {
    if (strategy == STRATEGY_LINEAR) {
        *f = 2.0 - t_norm * 1.5;
        *l = 1.5 - t_norm * 1.0;
        *m_param = 0.5 + t_norm * 0.3;
    } else if (strategy == STRATEGY_EXPONENTIAL) {
        *f = 2.0 * exp(-t_norm * 2.0); // Precomputed max_iter / 2
        *l = 1.5 * exp(-t_norm * 3.0); // Precomputed max_iter / 3
        *m_param = 0.5 * (1.0 + tanh(t_norm * 4.0)); // Precomputed max_iter / 4
    } else { // STRATEGY_CHAOTIC
        double x = 0.7;
        for (int i = 0; i < (t % 10); i++) {
            x = 4.0 * x * (1.0 - x);
        }
        *f = 1.5 + x * 0.5;
        *l = 1.0 + x * 0.5;
        *m_param = 0.3 + x * 0.4;
    }
}

// Update strategy selection probabilities based on historical success
void update_strategy_probabilities(double *strategy_probs, double *strategy_success) {
    double total_success = strategy_success[0] + strategy_success[1] + strategy_success[2] + 1e-10;
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] = strategy_success[i] / total_success;
        strategy_probs[i] = strategy_probs[i] < 0.1 ? 0.1 : (strategy_probs[i] > 0.9 ? 0.9 : strategy_probs[i]);
    }
    double sum_probs = strategy_probs[0] + strategy_probs[1] + strategy_probs[2];
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] /= sum_probs;
        strategy_success[i] *= 0.9;
    }
}

// Simulate penguin movement in huddle
void huddle_movement(Optimizer *opt, int index, int t, ObjectiveFunction objective_function, double *workspace) {
    static double strategy_probs[STRATEGY_COUNT] = {0.3333333333333333, 0.3333333333333333, 0.3333333333333333};
    static double strategy_success[STRATEGY_COUNT] = {0.0, 0.0, 0.0};
    
    double T_prime, R;
    compute_huddle_boundary(&T_prime, &R, t, opt->max_iter);

    // Select adaptation strategy
    double r = rand_double(0.0, 1.0);
    int strategy = r < strategy_probs[0] ? 0 : (r < strategy_probs[0] + strategy_probs[1] ? 1 : 2);

    double f, l, m_param;
    double t_norm = (double)t / opt->max_iter;
    adapt_parameters(&f, &l, &m_param, strategy, t_norm, t);

    // Compute distance and update position
    double S = m_param * exp(-t_norm / l) - exp(-t_norm);
    double *new_position = workspace;
    for (int j = 0; j < opt->dim; j++) {
        double D = fabs(f * rand_double(0.0, 1.0) * opt->best_solution.position[j] - opt->population[index].position[j]);
        new_position[j] = opt->population[index].position[j] + S * D * rand_double(0.0, 1.0);
        new_position[j] = new_position[j] < opt->bounds[2 * j] ? opt->bounds[2 * j] : 
                          new_position[j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : new_position[j];
    }

    // Evaluate new solution
    double new_fitness = objective_function(new_position);
    if (new_fitness < opt->population[index].fitness) {
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

    // Allocate workspace for huddle_movement
    double *workspace = (double *)malloc(opt->dim * sizeof(double));
    if (!workspace) {
        fprintf(stderr, "EPO_optimize: Memory allocation failed for workspace\n");
        return;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate all penguins in parallel
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == INFINITY) { // Only compute for unevaluated positions
                opt->population[i].fitness = objective_function(opt->population[i].position);
            }
            #pragma omp critical
            {
                if (opt->population[i].fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = opt->population[i].fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        opt->best_solution.position[j] = opt->population[i].position[j];
                    }
                }
            }
        }

        // Update penguin positions
        for (int i = 0; i < opt->population_size; i++) {
            huddle_movement(opt, i, iter, objective_function, workspace);
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(workspace);
}
