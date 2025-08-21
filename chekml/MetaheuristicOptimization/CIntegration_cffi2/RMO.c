#include "RMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize the population randomly within bounds
void rmo_initialize_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for each individual in the population
void rmo_evaluate_population(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Static Optimizer pointer for qsort comparison
static Optimizer *sort_opt;

// Comparison function for qsort
static int compare_indices(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    double fa = sort_opt->population[idx_a].fitness;
    double fb = sort_opt->population[idx_b].fitness;
    return (fa > fb) - (fa < fb);
}

// Sort population based on fitness (ascending order) without changing position pointers
void rmo_sort_population(Optimizer *opt) {
    // Allocate indices array on stack if population_size is small
    int indices[POPULATION_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Use qsort for faster sorting
    sort_opt = opt;
    qsort(indices, opt->population_size, sizeof(int), compare_indices);

    // Reorder position data and fitness in-place using cycle detection
    char visited[POPULATION_SIZE] = {0};
    double temp_position[opt->dim];
    for (int start = 0; start < opt->population_size; start++) {
        if (visited[start] || indices[start] == start) {
            continue;
        }
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
}

// Calculate the reference point and update positions in a single loop
void update_reference_point_and_positions(Optimizer *opt, double *reference_point) {
    // Initialize reference point
    for (int j = 0; j < opt->dim; j++) {
        reference_point[j] = 0.0;
    }

    // Compute reference point (mean) and update positions
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            reference_point[j] += position[j];
        }
    }

    // Finalize reference point and update positions
    for (int j = 0; j < opt->dim; j++) {
        reference_point[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double direction = reference_point[j] - position[j];
            position[j] += ALPHA * direction;
        }
    }

    enforce_bound_constraints(opt);
}

// Main Optimization Function
void RMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate memory for reference point on stack if dim is small
    double reference_point[opt->dim];

    // Initialize population
    rmo_initialize_population(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness for each individual
        rmo_evaluate_population(opt, objective_function);

        // Sort population based on fitness
        rmo_sort_population(opt);

        // Update best solution if a better one is found
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
        }

        // Update reference point and positions
        update_reference_point_and_positions(opt, reference_point);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
