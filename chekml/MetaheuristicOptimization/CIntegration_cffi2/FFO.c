#include "FFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator
#include <math.h>

/* Function to generate a random double between min and max */
double rand_double(double min, double max);

/* Initialize the fruit fly swarm and its axis */
void initialize_swarm(Optimizer *opt) {
    // 🌟 Allocate and initialize swarm axis (2D: X_axis, Y_axis)
    double *swarm_axis = (double *)malloc(2 * sizeof(double));
    if (!swarm_axis) {
        fprintf(stderr, "Memory allocation failed for swarm_axis\n");
        exit(1);
    }
    for (int j = 0; j < 2; j++) {
        swarm_axis[j] = rand_double(opt->bounds[2 * 0], opt->bounds[2 * 0 + 1]);  // Use first dimension bounds
    }

    // 🐝 Initialize swarm positions
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double random_value = SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            opt->population[i].position[j] = swarm_axis[j % 2] + random_value;  // Use 2D axis
        }
        opt->population[i].fitness = INFINITY;  // Initialize fitness
    }
    free(swarm_axis);  // Free temporary axis
    enforce_bound_constraints(opt);
}

/* Smell-based random search */
void smell_based_search(Optimizer *opt) {
    // 🌬️ Compute swarm axis as the current best solution
    double *swarm_axis = (double *)malloc(2 * sizeof(double));
    if (!swarm_axis) {
        fprintf(stderr, "Memory allocation failed for swarm_axis\n");
        exit(1);
    }
    swarm_axis[0] = opt->best_solution.position[0];
    swarm_axis[1] = opt->dim > 1 ? opt->best_solution.position[1] : opt->best_solution.position[0];

    // 🐝 Update each fruit fly's position
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double random_value = SMELL_RANGE * (2.0 * rand_double(0.0, 1.0) - 1.0);
            opt->population[i].position[j] = swarm_axis[j % 2] + random_value;
        }
    }
    free(swarm_axis);
    enforce_bound_constraints(opt);
}

/* Evaluate swarm's smell concentration (fitness) */
void evaluate_swarm(Optimizer *opt, double (*objective_function)(double *)) {
    // 🍎 Compute smell concentration for each fruit fly
    for (int i = 0; i < opt->population_size; i++) {
        double dist = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            dist += opt->population[i].position[j] * opt->population[i].position[j];
        }
        dist = sqrt(dist);
        double s = (dist != 0.0) ? 1.0 / dist : 1e-10;  // Avoid division by zero

        // Scale smell concentration to dimensions
        double *scaled_position = (double *)malloc(opt->dim * sizeof(double));
        if (!scaled_position) {
            fprintf(stderr, "Memory allocation failed for scaled_position\n");
            exit(1);
        }
        for (int j = 0; j < opt->dim; j++) {
            scaled_position[j] = s;
        }
        opt->population[i].fitness = objective_function(scaled_position);
        free(scaled_position);
    }
}

/* Vision-based update (move swarm to best solution) */
void vision_based_update(Optimizer *opt) {
    // 👀 Update swarm axis to best solution
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }

    // Update best solution if improved
    if (best_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = best_fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[best_idx].position[j];
        }
    }
}

/* Main Optimization Function */
void FFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // 🌱 Initialize random seed
    srand((unsigned int)time(NULL));

    // 🚀 Initialize swarm
    initialize_swarm(opt);
    evaluate_swarm(opt, objective_function);
    vision_based_update(opt);

    // 🔄 Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        smell_based_search(opt);
        evaluate_swarm(opt, objective_function);
        vision_based_update(opt);

        // 📈 Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
