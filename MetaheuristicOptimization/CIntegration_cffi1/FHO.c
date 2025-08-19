#include "FHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate a random double between min and max
double rand_double(double min, double max);

// Comparison function for qsort (doubles)
int compare_doubles(const void *a, const void *b) {
    double diff = *(double *)a - *(double *)b;
    return (diff > 0) - (diff < 0);
}

// Comparison function for qsort (solutions)
int compare_solutions(const void *a, const void *b) {
    double diff = ((Solution *)a)->fitness - ((Solution *)b)->fitness;
    return (diff > 0) - (diff < 0);
}

// Initialize the population randomly within bounds
void initialize_population_fho(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute Euclidean distance between two points
double euclidean_distance(double *a, double *b, int dim) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Assign prey to firehawks based on distance (optimized for speed)
void assign_prey_to_firehawks(Optimizer *opt, int num_firehawks, int *prey_assignments, int *prey_counts) {
    int prey_size = opt->population_size - num_firehawks;
    if (prey_size <= 0) {
        memset(prey_counts, 0, num_firehawks * sizeof(int));
        return;
    }

    double *distances = (double *)aligned_alloc(CACHE_LINE_SIZE, prey_size * sizeof(double));
    int *indices = (int *)aligned_alloc(CACHE_LINE_SIZE, prey_size * sizeof(int));
    memset(prey_counts, 0, num_firehawks * sizeof(int));

    for (int i = 0; i < num_firehawks; i++) {
        int remaining_prey = 0;
        for (int j = num_firehawks; j < opt->population_size; j++) {
            if (prey_assignments[j] == -1) {
                distances[remaining_prey] = euclidean_distance(opt->population[i].position, 
                                                              opt->population[j].position, opt->dim);
                indices[remaining_prey] = j;
                remaining_prey++;
            }
        }

        if (remaining_prey == 0) {
            prey_counts[i] = 0;
            continue;
        }

        // Sort distances using qsort
        qsort(distances, remaining_prey, sizeof(double), compare_doubles);

        // Assign random number of prey
        int num_prey = rand() % remaining_prey + 1;
        prey_counts[i] = num_prey;

        // Assign nearest prey
        for (int j = 0; j < num_prey; j++) {
            prey_assignments[indices[j]] = i;
        }
    }

    free(distances);
    free(indices);
}

// Main optimization function (optimized for extreme speed)
void FHO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned)time(NULL));
    initialize_population_fho(opt);

    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    // Pre-allocate all memory
    double *global_safe_point = (double *)aligned_alloc(CACHE_LINE_SIZE, opt->dim * sizeof(double));
    double *temp_buffer = (double *)aligned_alloc(CACHE_LINE_SIZE, opt->dim * sizeof(double));
    int *prey_assignments = (int *)aligned_alloc(CACHE_LINE_SIZE, opt->population_size * sizeof(int));
    int *prey_counts = (int *)aligned_alloc(CACHE_LINE_SIZE, opt->population_size * sizeof(int));
    double **new_positions = (double **)aligned_alloc(CACHE_LINE_SIZE, 3 * opt->population_size * sizeof(double *));
    Solution *solutions = (Solution *)aligned_alloc(CACHE_LINE_SIZE, 3 * opt->population_size * sizeof(Solution));

    // Pre-allocate new positions
    for (int i = 0; i < 3 * opt->population_size; i++) {
        new_positions[i] = (double *)aligned_alloc(CACHE_LINE_SIZE, opt->dim * sizeof(double));
        solutions[i].position = new_positions[i];
    }

    int iter = 0, fes = opt->population_size;

    while (fes < opt->max_iter) {
        iter++;

        // Compute global safe point
        memset(global_safe_point, 0, opt->dim * sizeof(double));
        for (int i = 0; i < opt->population_size; i++) {
            #pragma omp simd
            for (int j = 0; j < opt->dim; j++) {
                global_safe_point[j] += opt->population[i].position[j];
            }
        }
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            global_safe_point[j] /= opt->population_size;
        }

        // Determine number of firehawks
        int max_firehawks = (int)(ceil(opt->population_size * MAX_FIREHAWKS_RATIO)) + 1;
        int num_firehawks = MIN_FIREHAWKS + (rand() % (max_firehawks - MIN_FIREHAWKS + 1));

        // Initialize prey assignments
        memset(prey_assignments, -1, opt->population_size * sizeof(int));
        memset(prey_counts, 0, num_firehawks * sizeof(int));

        // Assign prey to firehawks
        assign_prey_to_firehawks(opt, num_firehawks, prey_assignments, prey_counts);

        int new_pop_size = 0;

        // Update firehawks
        for (int i = 0; i < num_firehawks; i++) {
            int other_fh_idx = rand() % num_firehawks;
            update_firehawk_position(opt, opt->population[i].position, 
                                    opt->population[other_fh_idx].position, 
                                    new_positions[new_pop_size]);
            solutions[new_pop_size].fitness = objective_function(new_positions[new_pop_size]);
            new_pop_size++;
            fes++;
        }

        // Update prey
        for (int i = 0; i < num_firehawks; i++) {
            if (prey_counts[i] == 0) continue;

            // Compute local safe point
            memset(temp_buffer, 0, opt->dim * sizeof(double));
            int count = 0;
            for (int j = num_firehawks; j < opt->population_size; j++) {
                if (prey_assignments[j] == i) {
                    #pragma omp simd
                    for (int k = 0; k < opt->dim; k++) {
                        temp_buffer[k] += opt->population[j].position[k];
                    }
                    count++;
                }
            }
            if (count > 0) {
                #pragma omp simd
                for (int k = 0; k < opt->dim; k++) {
                    temp_buffer[k] /= count;
                }
            }

            // Update each prey
            for (int j = num_firehawks; j < opt->population_size; j++) {
                if (prey_assignments[j] == i) {
                    update_prey_position(opt, opt->population[j].position, 
                                        opt->population[i].position, temp_buffer, 
                                        global_safe_point, new_positions[new_pop_size], 
                                        new_positions[new_pop_size + 1]);
                    solutions[new_pop_size].fitness = objective_function(new_positions[new_pop_size]);
                    solutions[new_pop_size + 1].fitness = objective_function(new_positions[new_pop_size + 1]);
                    new_pop_size += 2;
                    fes += 2;
                }
            }
        }

        // Sort new population
        qsort(solutions, new_pop_size, sizeof(Solution), compare_solutions);

        // Update main population
        for (int i = 0; i < opt->population_size && i < new_pop_size; i++) {
            memcpy(opt->population[i].position, solutions[i].position, opt->dim * sizeof(double));
            opt->population[i].fitness = solutions[i].fitness;
            if (solutions[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = solutions[i].fitness;
                memcpy(opt->best_solution.position, solutions[i].position, opt->dim * sizeof(double));
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", iter, opt->best_solution.fitness);
    }

    // Clean up
    for (int i = 0; i < 3 * opt->population_size; i++) {
        free(new_positions[i]);
    }
    free(new_positions);
    free(solutions);
    free(prey_assignments);
    free(prey_counts);
    free(global_safe_point);
    free(temp_buffer);
}
