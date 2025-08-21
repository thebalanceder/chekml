#include "CulA.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define ENABLE_LOGGING 0  // Set to 1 to enable iteration log output

// 🔒 Safe cleanup macro
#define CLEANUP_CULTURE(c) do { \
    free(c.situational.position); \
    free(c.normative.min); \
    free(c.normative.max); \
    free(c.normative.L); \
    free(c.normative.U); \
    free(c.normative.size); \
} while (0)

// 🎲 Generate random double between min and max
double rand_double(double min, double max);

// 🎲 Generate random normal variable (Box-Muller transform)
double rand_normal_cula(double mean, double stddev) {
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
    int dim = opt->dim;

    culture->situational.position = calloc(dim, sizeof(double));
    culture->normative.min = malloc(dim * sizeof(double));
    culture->normative.max = malloc(dim * sizeof(double));
    culture->normative.L = malloc(dim * sizeof(double));
    culture->normative.U = malloc(dim * sizeof(double));
    culture->normative.size = malloc(dim * sizeof(double));

    if (!culture->situational.position || !culture->normative.min ||
        !culture->normative.max || !culture->normative.L ||
        !culture->normative.U || !culture->normative.size) {
        fprintf(stderr, "🚫 Error: Memory allocation failed in initialize_culture\n");
        CLEANUP_CULTURE((*culture));
        memset(culture, 0, sizeof(Culture));
        return;
    }

    culture->situational.cost = INFINITY;
    for (int j = 0; j < dim; j++) {
        culture->normative.min[j] = INFINITY;
        culture->normative.max[j] = -INFINITY;
        culture->normative.L[j] = INFINITY;
        culture->normative.U[j] = INFINITY;
        culture->normative.size[j] = 0.0;
    }
}

// 🔢 Struct for sorting by cost
typedef struct {
    int index;
    double cost;
} IndexedCost;

int compare_costs(const void *a, const void *b) {
    double diff = ((IndexedCost *)a)->cost - ((IndexedCost *)b)->cost;
    return (diff > 0) - (diff < 0);
}

// 🌍 Adjust Culture Based on Top n_accept Individuals
void adjust_culture(Optimizer *opt, Culture *culture, int n_accept) {
    int dim = opt->dim;
    int pop_size = opt->population_size;

    IndexedCost *ranked = malloc(pop_size * sizeof(IndexedCost));
    if (!ranked) {
        fprintf(stderr, "🚫 Error: Memory allocation failed in adjust_culture\n");
        return;
    }

    for (int i = 0; i < pop_size; i++) {
        ranked[i].index = i;
        ranked[i].cost = opt->population[i].fitness;
    }

    qsort(ranked, pop_size, sizeof(IndexedCost), compare_costs);

    for (int i = 0; i < n_accept && i < pop_size; i++) {
        int idx = ranked[i].index;
        double cost = opt->population[idx].fitness;
        double *pos = opt->population[idx].position;

        if (cost < culture->situational.cost) {
            culture->situational.cost = cost;
            memcpy(culture->situational.position, pos, dim * sizeof(double));
        }

        for (int j = 0; j < dim; j++) {
            if (pos[j] < culture->normative.min[j] || cost < culture->normative.L[j]) {
                culture->normative.min[j] = pos[j];
                culture->normative.L[j] = cost;
            }
            if (pos[j] > culture->normative.max[j] || cost < culture->normative.U[j]) {
                culture->normative.max[j] = pos[j];
                culture->normative.U[j] = cost;
            }
            culture->normative.size[j] = culture->normative.max[j] - culture->normative.min[j];
        }
    }

    free(ranked);
}

// 🌍 Apply Cultural Influence to Update Population
void influence_culture(Optimizer *opt, Culture *culture) {
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double *situational = culture->situational.position;
    double *size = culture->normative.size;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double sigma = ALPHA_SCALING * size[j];
            double dx = rand_normal_cula(0.0, sigma);
            dx = (pos[j] < situational[j]) ? fabs(dx) : (pos[j] > situational[j]) ? -fabs(dx) : 0.0;
            pos[j] += dx;
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
    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;

    if (!opt->population || !opt->bounds || dim <= 0 || pop_size <= 0 || max_iter <= 0) {
        fprintf(stderr, "🚫 Error: Invalid optimizer parameters\n");
        return;
    }

    srand((unsigned int)time(NULL));

    Culture culture = {0};
    initialize_culture(opt, &culture);
    if (!culture.situational.position) return;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(pos);
    }

    int n_accept = (int)(ACCEPTANCE_RATIO * pop_size);
    if (n_accept < 1) n_accept = 1;

    adjust_culture(opt, &culture, n_accept);

    if (!opt->best_solution.position) {
        CLEANUP_CULTURE(culture);
        return;
    }

    opt->best_solution.fitness = culture.situational.cost;
    memcpy(opt->best_solution.position, culture.situational.position, dim * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        influence_culture(opt, &culture);
        for (int i = 0; i < pop_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        adjust_culture(opt, &culture, n_accept);

        if (culture.situational.cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = culture.situational.cost;
            memcpy(opt->best_solution.position, culture.situational.position, dim * sizeof(double));
        }

        #if ENABLE_LOGGING
        printf("🔄 Iteration %d: Best Cost = %.8f\n", iter + 1, opt->best_solution.fitness);
        #endif
    }

    CLEANUP_CULTURE(culture);
}

