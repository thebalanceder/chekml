#include "FFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

#define ENABLE_LOGGING 0
#define EPSILON 1e-10

// 🎲 Generate random double between min and max
double rand_double(double min, double max);

// 🌱 Initialize fruit fly population and swarm axis
void initialize_swarm(Optimizer *opt) {
    int dim = opt->dim;
    double x_axis = rand_double(opt->bounds[0], opt->bounds[1]);
    double y_axis = (dim > 1) ? rand_double(opt->bounds[2], opt->bounds[3]) : x_axis;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < dim; j++) {
            double offset = SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            double base = (j % 2 == 0) ? x_axis : y_axis;
            opt->population[i].position[j] = base + offset;
        }
        opt->population[i].fitness = INFINITY;
    }

    enforce_bound_constraints(opt);
}

// 🌀 Smell-based local random search
void smell_based_search(Optimizer *opt) {
    int dim = opt->dim;
    double x_axis = opt->best_solution.position[0];
    double y_axis = (dim > 1) ? opt->best_solution.position[1] : x_axis;

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < dim; j++) {
            double offset = SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            double base = (j % 2 == 0) ? x_axis : y_axis;
            opt->population[i].position[j] = base + offset;
        }
    }

    enforce_bound_constraints(opt);
}

// 👃 Evaluate smell concentration (fitness)
void evaluate_swarm(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *shared_position = (double *)malloc(dim * sizeof(double));
    if (!shared_position) {
        fprintf(stderr, "🚫 Memory allocation failed in evaluate_swarm\n");
        exit(1);
    }

    for (int i = 0; i < opt->population_size; i++) {
        double norm = 0.0;
        for (int j = 0; j < dim; j++) {
            double x = opt->population[i].position[j];
            norm += x * x;
        }

        double s = (norm > EPSILON) ? 1.0 / sqrt(norm) : 1.0 / sqrt(EPSILON);

        for (int j = 0; j < dim; j++) {
            shared_position[j] = s;  // scalar smell replicated across all dimensions
        }

        opt->population[i].fitness = objective_function(shared_position);
    }

    free(shared_position);
}

// 👁️ Select best individual in swarm and update global best
void vision_based_update(Optimizer *opt) {
    int best_idx = 0;
    double best_fit = opt->population[0].fitness;

    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fit) {
            best_fit = opt->population[i].fitness;
            best_idx = i;
        }
    }

    if (best_fit < opt->best_solution.fitness) {
        opt->best_solution.fitness = best_fit;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[best_idx].position[j];
        }
    }
}

// 🚀 Main FFO Optimization
void FFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds) {
        fprintf(stderr, "🚫 Invalid input to FFO_optimize\n");
        return;
    }

    srand((unsigned int)time(NULL));

    initialize_swarm(opt);
    evaluate_swarm(opt, objective_function);
    vision_based_update(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        smell_based_search(opt);
        evaluate_swarm(opt, objective_function);
        vision_based_update(opt);

        #if ENABLE_LOGGING
        printf("Iteration %d: Best Value = %.8f\n", iter + 1, opt->best_solution.fitness);
        #endif
    }
}

