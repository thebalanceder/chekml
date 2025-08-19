#include "SSA.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize squirrel positions and tree types
void initialize_squirrels(Optimizer *opt, ObjectiveFunction objective_function, int *tree_types) {
    int i, j;

    // Initialize squirrels
    for (i = 0; i < opt->population_size; i++) {
        for (j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        tree_types[i] = (rand() % 3) + 1; // Assign tree type: 1 (acorn), 2 (normal), 3 (hickory)
    }
    enforce_bound_constraints(opt);

    // Find initial best solution
    int min_idx = 0;
    for (i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[min_idx].fitness;
    for (j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[min_idx].position[j];
    }
}

// Update position of a single squirrel
void update_squirrel_position(Optimizer *opt, int idx, ObjectiveFunction objective_function, int *tree_types) {
    int j;
    double gliding_distance = MIN_GLIDING_DISTANCE + (MAX_GLIDING_DISTANCE - MIN_GLIDING_DISTANCE) * rand_double(0.0, 1.0);
    double pulse_flying_rate = rand_double(0.0, 1.0);

    // Temporary array for velocity
    double *velocity = (double *)calloc(opt->dim, sizeof(double));
    if (!velocity) {
        fprintf(stderr, "Memory allocation failed for velocity\n");
        return;
    }

    // Update velocity based on tree type
    if (tree_types[idx] == 1) { // Acorn tree
        for (j = 0; j < opt->dim; j++) {
            velocity[j] += gliding_distance * GLIDING_CONSTANT * 
                          (opt->population[idx].position[j] - opt->best_solution.position[j]) * 1;
        }
    } else if (tree_types[idx] == 2) { // Normal tree
        for (j = 0; j < opt->dim; j++) {
            velocity[j] += gliding_distance * GLIDING_CONSTANT * 
                          (opt->population[idx].position[j] - opt->best_solution.position[j]) * 2;
        }
    } else { // Hickory tree
        for (j = 0; j < opt->dim; j++) {
            velocity[j] += gliding_distance * GLIDING_CONSTANT * 
                          (opt->population[idx].position[j] - opt->best_solution.position[j]) * 3;
        }
    }

    // Update position
    for (j = 0; j < opt->dim; j++) {
        opt->population[idx].position[j] += velocity[j];
    }
    enforce_bound_constraints(opt);

    // Random flying condition
    if (rand_double(0.0, 1.0) > pulse_flying_rate) {
        double eps = -1.0 + (1.0 - (-1.0)) * rand_double(0.0, 1.0);
        double mean_A = 0.0;
        for (j = 0; j < opt->population_size; j++) {
            mean_A += rand_double(0.0, 1.0);
        }
        mean_A /= opt->population_size;
        for (j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] = opt->best_solution.position[j] + eps * mean_A;
        }
        enforce_bound_constraints(opt);
    }

    // Evaluate new fitness
    double new_fitness = objective_function(opt->population[idx].position);
    if (new_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = new_fitness;
        for (j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[idx].position[j];
        }
    }

    opt->population[idx].fitness = new_fitness;
    free(velocity);
}

// Main Optimization Function
void SSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int iter, i;
    // Seed random number generator
    srand(time(NULL));

    // Allocate array for tree types
    int *tree_types = (int *)malloc(opt->population_size * sizeof(int));
    if (!tree_types) {
        fprintf(stderr, "Memory allocation failed for tree_types\n");
        return;
    }

    // Initialize squirrels
    initialize_squirrels(opt, objective_function, tree_types);

    // Main optimization loop
    for (iter = 0; iter < opt->max_iter; iter++) {
        for (i = 0; i < opt->population_size; i++) {
            update_squirrel_position(opt, i, objective_function, tree_types);
        }
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up tree_types
    free(tree_types);
}
