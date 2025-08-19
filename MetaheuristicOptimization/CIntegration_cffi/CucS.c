#include "CucS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Random double generator
double rand_double(double min, double max);

// Initialize random seed once
static void init_rand_seed() {
    static int initialized = 0;
    if (!initialized) {
        srand((unsigned int)time(NULL));
        initialized = 1;
    }
}

// Initialize nests
void initialize_nests(Optimizer *restrict opt) {
    init_rand_seed();
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = min + (max - min) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness
void evaluate_nests(Optimizer *restrict opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Generate new solutions via Levy flights
void get_cuckoos(Optimizer *restrict opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict s = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double u = rand_double(-1.0, 1.0) * CS_SIGMA;
            double v = rand_double(-1.0, 1.0);
            double step = v != 0.0 ? u / pow(fabs(v), 1.0 / CS_BETA) : u;
            double stepsize = CS_STEP_SCALE * step * (s[j] - opt->best_solution.position[j]);
            s[j] += stepsize * rand_double(-1.0, 1.0);
        }
    }
    enforce_bound_constraints(opt);
}

// Replace some nests
void empty_nests(Optimizer *restrict opt) {
    int n = opt->population_size;
    double stepsize[n][opt->dim];
    int idx[n], idx2[n];

    // Generate random permutations and step sizes
    for (int i = 0; i < n; i++) {
        idx[i] = i;
        idx2[i] = i;
        for (int j = 0; j < opt->dim; j++) {
            stepsize[i][j] = (rand_double(0.0, 1.0) > CS_PA) ? 
                             rand_double(0.0, 1.0) * (opt->population[i].position[j] - opt->population[rand() % n].position[j]) : 0.0;
        }
    }

    // Shuffle indices
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = idx[i]; idx[i] = idx[j]; idx[j] = temp;
        j = rand() % (i + 1);
        temp = idx2[i]; idx2[i] = idx2[j]; idx2[j] = temp;
    }

    // Update nests
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += stepsize[idx[i]][j];
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void CucS_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    opt->population_size = CS_POPULATION_SIZE;
    opt->max_iter = CS_MAX_ITER;
    initialize_nests(opt);
    evaluate_nests(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        get_cuckoos(opt);
        evaluate_nests(opt, objective_function);
        empty_nests(opt);
        evaluate_nests(opt, objective_function);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    printf("Total evaluations=%d\n", opt->max_iter * opt->population_size * 2);
}
