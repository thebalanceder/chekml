#include "RMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize the population randomly within bounds
void initialize_population_rmo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for each individual in the population
void evaluate_population_rmo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Sort population based on fitness (ascending order) without changing position pointers
void rmo_sort_population(Optimizer *opt) {
    // Create an array of indices
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "rmo_sort_population: Memory allocation failed for indices\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Bubble sort indices based on fitness
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[indices[j]].fitness > opt->population[indices[j + 1]].fitness) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // Reorder position data and fitness in-place
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "rmo_sort_population: Memory allocation failed for temp_position\n");
        free(indices);
        return;
    }

    // Apply permutation to population using cycle detection to minimize swaps
    char *visited = (char *)calloc(opt->population_size, sizeof(char));
    if (!visited) {
        fprintf(stderr, "rmo_sort_population: Memory allocation failed for visited\n");
        free(temp_position);
        free(indices);
        return;
    }

    for (int start = 0; start < opt->population_size; start++) {
        if (visited[start] || indices[start] == start) {
            continue;
        }
        // Follow the cycle
        int current = start;
        double temp_fitness = opt->population[start].fitness;
        memcpy(temp_position, opt->population[start].position, opt->dim * sizeof(double));
        while (!visited[current]) {
            visited[current] = 1;
            int next = indices[current];
            if (next != start) {
                opt->population[current].fitness = opt->population[next].fitness;
                memcpy(opt->population[current].position, opt->population[next].position, opt->dim * sizeof(double));
            } else {
                opt->population[current].fitness = temp_fitness;
                memcpy(opt->population[current].position, temp_position, opt->dim * sizeof(double));
            }
            current = next;
        }
    }

    // Free allocated memory
    free(visited);
    free(temp_position);
    free(indices);
}

// Calculate the reference point as the mean of the population
void update_reference_point(Optimizer *opt, double *reference_point) {
    for (int j = 0; j < opt->dim; j++) {
        reference_point[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            reference_point[j] += opt->population[i].position[j];
        }
        reference_point[j] /= opt->population_size;
    }
}

// Update positions towards the reference point
void rmo_update_positions(Optimizer *opt, double *reference_point) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double direction = reference_point[j] - opt->population[i].position[j];
            opt->population[i].position[j] += ALPHA * direction;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void RMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory for reference point
    double *reference_point = (double *)malloc(opt->dim * sizeof(double));
    if (!reference_point) {
        fprintf(stderr, "RMO_optimize: Memory allocation failed for reference_point\n");
        return;
    }

    // Initialize population
    initialize_population_rmo(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        evaluate_population_rmo(opt, objective_function);

        // Sort population based on fitness
        rmo_sort_population(opt);

        // Update best solution if a better one is found
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
        }

        // Update reference point
        update_reference_point(opt, reference_point);

        // Update population positions
        rmo_update_positions(opt, reference_point);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free allocated memory
    free(reference_point);
}
