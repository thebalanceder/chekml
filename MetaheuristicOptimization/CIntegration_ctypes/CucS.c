#include "CucS.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand()
#include <time.h>    // For time() to seed random generator
#include <math.h>    // For mathematical operations

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize nests (population) randomly within bounds
void initialize_nests(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all nests
void evaluate_nests(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Generate new solutions via Levy flights
void get_cuckoos(Optimizer *opt) {
    double sigma = pow(CS_GAMMA_BETA * sin(CS_PI * CS_BETA / 2) /
                       (CS_GAMMA_HALF_BETA * CS_BETA * pow(2, (CS_BETA - 1) / 2)), 1.0 / CS_BETA);

    for (int i = 0; i < opt->population_size; i++) {
        double *s = opt->population[i].position;
        double u[opt->dim], v[opt->dim], step[opt->dim], stepsize[opt->dim];

        // Generate random numbers for Levy flight
        for (int j = 0; j < opt->dim; j++) {
            u[j] = rand_double(-1.0, 1.0) * sigma;
            v[j] = rand_double(-1.0, 1.0);
            step[j] = v[j] != 0 ? u[j] / pow(fabs(v[j]), 1.0 / CS_BETA) : u[j];
            stepsize[j] = CS_STEP_SCALE * step[j] * (s[j] - opt->best_solution.position[j]);
            s[j] += stepsize[j] * rand_double(-1.0, 1.0);
        }
    }
    enforce_bound_constraints(opt);
}

// Replace some nests based on discovery probability
void empty_nests(Optimizer *opt) {
    int n = opt->population_size;
    double K[n][opt->dim];
    int idx[n], idx2[n];

    // Generate random mask K
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < opt->dim; j++) {
            K[i][j] = rand_double(0.0, 1.0) > CS_PA ? 1.0 : 0.0;
        }
    }

    // Generate random permutations
    for (int i = 0; i < n; i++) {
        idx[i] = i;
        idx2[i] = i;
    }
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = idx[i];
        idx[i] = idx[j];
        idx[j] = temp;
        j = rand() % (i + 1);
        temp = idx2[i];
        idx2[i] = idx2[j];
        idx2[j] = temp;
    }

    // Update nests
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double stepsize = rand_double(0.0, 1.0) * (opt->population[idx[i]].position[j] - opt->population[idx2[i]].position[j]);
            opt->population[i].position[j] += stepsize * K[i][j];
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void CucS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_nests(opt);
    evaluate_nests(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Generate new solutions via Levy flights
        get_cuckoos(opt);
        evaluate_nests(opt, objective_function);

        // Discovery and randomization
        empty_nests(opt);
        evaluate_nests(opt, objective_function);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    printf("Total number of iterations=%d\n", opt->max_iter * opt->population_size * 2);
}
