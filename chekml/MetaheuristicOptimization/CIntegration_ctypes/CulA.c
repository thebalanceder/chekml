#include "CulA.h"
#include <stdlib.h>  // ✅ For rand() and srand()
#include <time.h>    // ✅ For time() to seed random generator
#include <math.h>    // ✅ For mathematical functions
#include <stdio.h>   // ✅ For error logging

// 🎲 Generate random double between min and max
double rand_double(double min, double max);

// 🎲 Generate random normal variable (Box-Muller transform)
double rand_normal(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }

    has_spare = 1;
    double u = rand_double(0.0, 1.0);
    double v = rand_double(0.0, 1.0);
    double s = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    spare = sqrt(-2.0 * log(u)) * sin(2.0 * M_PI * v);
    return mean + stddev * s;
}

// 🌍 Initialize Culture Components
void initialize_culture(Optimizer *opt, Culture *culture) {
    if (!opt || !culture) {
        fprintf(stderr, "🚫 Error: Null optimizer or culture\n");
        return;
    }

    int dim = opt->dim;
    culture->situational.position = (double *)malloc(dim * sizeof(double));
    culture->normative.min = (double *)malloc(dim * sizeof(double));
    culture->normative.max = (double *)malloc(dim * sizeof(double));
    culture->normative.L = (double *)malloc(dim * sizeof(double));
    culture->normative.U = (double *)malloc(dim * sizeof(double));
    culture->normative.size = (double *)malloc(dim * sizeof(double));

    if (!culture->situational.position || !culture->normative.min || 
        !culture->normative.max || !culture->normative.L || 
        !culture->normative.U || !culture->normative.size) {
        fprintf(stderr, "🚫 Error: Memory allocation failed\n");
        free(culture->situational.position);
        free(culture->normative.min);
        free(culture->normative.max);
        free(culture->normative.L);
        free(culture->normative.U);
        free(culture->normative.size);
        culture->situational.position = NULL;
        culture->normative.min = NULL;
        culture->normative.max = NULL;
        culture->normative.L = NULL;
        culture->normative.U = NULL;
        culture->normative.size = NULL;
        return;
    }

    culture->situational.cost = INFINITY;
    for (int j = 0; j < dim; j++) {
        culture->situational.position[j] = 0.0;
        culture->normative.min[j] = INFINITY;
        culture->normative.max[j] = -INFINITY;
        culture->normative.L[j] = INFINITY;
        culture->normative.U[j] = INFINITY;
        culture->normative.size[j] = 0.0;
    }
}

// 🌍 Adjust Culture Based on Top n_accept Individuals
void adjust_culture(Optimizer *opt, Culture *culture, int n_accept) {
    if (!opt || !culture || !opt->population || n_accept <= 0) {
        fprintf(stderr, "🚫 Error: Invalid inputs in adjust_culture\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;

    // 📊 Allocate temporary arrays for sorting
    double *costs = (double *)malloc(pop_size * sizeof(double));
    int *indices = (int *)malloc(pop_size * sizeof(int));
    if (!costs || !indices) {
        fprintf(stderr, "🚫 Error: Memory allocation failed\n");
        free(costs);
        free(indices);
        return;
    }

    // 📊 Collect fitness values and indices
    for (int i = 0; i < pop_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "🚫 Error: Null population position at index %d\n", i);
            free(costs);
            free(indices);
            return;
        }
        costs[i] = opt->population[i].fitness;
        indices[i] = i;
    }

    // 📊 Bubble sort to find top n_accept individuals
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = 0; j < pop_size - i - 1; j++) {
            if (costs[indices[j]] > costs[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // 🧠 Update situational and normative knowledge
    for (int i = 0; i < n_accept && i < pop_size; i++) {
        int idx = indices[i];
        double cost = costs[idx];

        // Situational component
        if (cost < culture->situational.cost) {
            culture->situational.cost = cost;
            for (int j = 0; j < dim; j++) {
                culture->situational.position[j] = opt->population[idx].position[j];
            }
        }

        // Normative component
        for (int j = 0; j < dim; j++) {
            double pos = opt->population[idx].position[j];
            if (pos < culture->normative.min[j] || cost < culture->normative.L[j]) {
                culture->normative.min[j] = pos;
                culture->normative.L[j] = cost;
            }
            if (pos > culture->normative.max[j] || cost < culture->normative.U[j]) {
                culture->normative.max[j] = pos;
                culture->normative.U[j] = cost;
            }
            culture->normative.size[j] = culture->normative.max[j] - culture->normative.min[j];
        }
    }

    free(costs);
    free(indices);
}

// 🌍 Apply Cultural Influence to Update Population
void influence_culture(Optimizer *opt, Culture *culture) {
    if (!opt || !culture || !opt->population || !culture->situational.position) {
        fprintf(stderr, "🚫 Error: Invalid inputs in influence_culture\n");
        return;
    }

    int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "🚫 Error: Null population position at index %d\n", i);
            return;
        }
        for (int j = 0; j < dim; j++) {
            double sigma = ALPHA_SCALING * culture->normative.size[j];
            double dx = rand_normal(0.0, sigma);
            if (opt->population[i].position[j] < culture->situational.position[j]) {
                dx = fabs(dx);
            } else if (opt->population[i].position[j] > culture->situational.position[j]) {
                dx = -fabs(dx);
            }
            opt->population[i].position[j] += dx;
        }
    }
    enforce_bound_constraints(opt);
}

// 🚀 Main Optimization Function
void CulA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    if (!opt_void || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs to CulA_optimize\n");
        return;
    }

    Optimizer *opt = (Optimizer *)opt_void;

    if (!opt->population || !opt->bounds) {
        fprintf(stderr, "🚫 Error: Null population or bounds\n");
        return;
    }

    if (opt->dim <= 0 || opt->population_size <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "🚫 Error: Invalid optimizer parameters\n");
        return;
    }

    // 🎲 Initialize random seed
    srand((unsigned int)time(NULL));

    // 🧠 Initialize culture
    Culture culture = {0}; // Zero-initialize to ensure clean state
    initialize_culture(opt, &culture);
    if (!culture.situational.position) {
        fprintf(stderr, "🚫 Error: Culture initialization failed\n");
        return;
    }

    // 🌍 Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "🚫 Error: Null population position at index %d\n", i);
            // Clean up culture
            free(culture.situational.position);
            free(culture.normative.min);
            free(culture.normative.max);
            free(culture.normative.L);
            free(culture.normative.U);
            free(culture.normative.size);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] +
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // 📊 Calculate number of accepted individuals
    int n_accept = (int)(ACCEPTANCE_RATIO * opt->population_size);
    if (n_accept < 1) n_accept = 1;

    // 🧠 Initialize culture with initial population
    adjust_culture(opt, &culture, n_accept);

    // 🌟 Set initial best solution
    if (!opt->best_solution.position) {
        fprintf(stderr, "🚫 Error: Null best_solution.position\n");
        // Clean up culture
        free(culture.situational.position);
        free(culture.normative.min);
        free(culture.normative.max);
        free(culture.normative.L);
        free(culture.normative.U);
        free(culture.normative.size);
        return;
    }
    opt->best_solution.fitness = culture.situational.cost;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = culture.situational.position[j];
    }

    // 🔄 Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        influence_culture(opt, &culture);
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        adjust_culture(opt, &culture, n_accept);
        if (culture.situational.cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = culture.situational.cost;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = culture.situational.position[j];
            }
        }
        printf("🔄 Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // 🧹 Clean up culture memory
    free(culture.situational.position);
    free(culture.normative.min);
    free(culture.normative.max);
    free(culture.normative.L);
    free(culture.normative.U);
    free(culture.normative.size);
}
