#include "RMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Ensure cache line alignment for performance-critical arrays
#define CACHE_LINE_ALIGN __attribute__((aligned(64)))

// Maximum dimension for stack-allocated arrays
#define MAX_DIM 64

// Function to generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize the population randomly within bounds
void rmo_initialize_population(Optimizer *opt) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            position[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for each individual in parallel
void rmo_evaluate_population(Optimizer *opt, double (*objective_function)(double *)) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Static Optimizer pointer for qsort comparison
static Optimizer *sort_opt;

// Comparison function for qsort
static inline int compare_indices(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    double fa = sort_opt->population[idx_a].fitness;
    double fb = sort_opt->population[idx_b].fitness;
    return (fa > fb) - (fa < fb);
}

// Sort population based on fitness (ascending order) without changing position pointers
void rmo_sort_population(Optimizer *opt) {
    // Stack-allocated indices array with cache line alignment
    int indices[POPULATION_SIZE] CACHE_LINE_ALIGN;
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }

    // Use qsort for efficient sorting
    sort_opt = opt;
    qsort(indices, opt->population_size, sizeof(int), compare_indices);

    // Reorder position data and fitness in-place using cycle detection
    char visited[POPULATION_SIZE] CACHE_LINE_ALIGN = {0};
    double temp_position[MAX_DIM] CACHE_LINE_ALIGN;
    if (opt->dim > MAX_DIM) {
        fprintf(stderr, "rmo_sort_population: Dimension %d exceeds MAX_DIM %d\n", opt->dim, MAX_DIM);
        return;
    }
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

// Calculate the reference point and update positions with vectorization
void update_reference_point_and_positions(Optimizer *opt, double *reference_point) {
    // Initialize reference point
    #pragma omp parallel for
    for (int j = 0; j < opt->dim; j++) {
        reference_point[j] = 0.0;
    }

    // Precompute inverse population size
    const double inv_pop_size = 1.0 / opt->population_size;

    // Compute reference point (mean) in parallel
    double temp_sums[MAX_DIM] CACHE_LINE_ALIGN = {0.0};
    if (opt->dim > MAX_DIM) {
        fprintf(stderr, "update_reference_point_and_positions: Dimension %d exceeds MAX_DIM %d\n", opt->dim, MAX_DIM);
        return;
    }
    #pragma omp parallel for reduction(+:temp_sums[:opt->dim])
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            temp_sums[j] += position[j];
        }
    }

    // Finalize reference point
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        reference_point[j] = temp_sums[j] * inv_pop_size;
    }

    // Update positions in parallel with vectorization
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            double direction = reference_point[j] - position[j];
            position[j] += ALPHA * direction;
        }
    }

    enforce_bound_constraints(opt);
}

// Main Optimization Function
void RMO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Stack-allocated reference point with cache line alignment
    double reference_point[MAX_DIM] CACHE_LINE_ALIGN;
    if (opt->dim > MAX_DIM) {
        fprintf(stderr, "RMO_optimize: Dimension %d exceeds MAX_DIM %d\n", opt->dim, MAX_DIM);
        return;
    }

    // Initialize population
    rmo_initialize_population(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        rmo_evaluate_population(opt, objective_function);

        // Sort population
        rmo_sort_population(opt);

        // Update best solution
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            #pragma omp simd
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
