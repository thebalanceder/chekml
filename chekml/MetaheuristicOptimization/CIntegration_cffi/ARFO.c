/* ARFO.c - Optimized Implementation file for Artificial Root Foraging Optimization */
#include "ARFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Comparison function for qsort on doubles
int compare_doubles_arfo(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

// Comparison function for qsort on fitness with indices
int compare_fitness_index(const void *a, const void *b) {
    double fa = ((FitnessIndex *)a)->fitness;
    double fb = ((FitnessIndex *)b)->fitness;
    return (fa > fb) - (fa < fb);
}

// Function to compute auxin concentration based on fitness
void calculate_auxin_concentration(Optimizer *opt, double *fitness, double *auxin) {
    double f_min = fitness[0], f_max = fitness[0];
    for (int i = 1; i < opt->population_size; i++) {
        if (fitness[i] < f_min) f_min = fitness[i];
        if (fitness[i] > f_max) f_max = fitness[i];
    }

    double sum_normalized = 0.0;
    if (f_max == f_min) {
        for (int i = 0; i < opt->population_size; i++) {
            auxin[i] = 1.0 / opt->population_size;
            sum_normalized += auxin[i];
        }
    } else {
        for (int i = 0; i < opt->population_size; i++) {
            auxin[i] = (fitness[i] - f_min) / (f_max - f_min);
            sum_normalized += auxin[i];
        }
    }

    for (int i = 0; i < opt->population_size; i++) {
        auxin[i] = auxin[i] / sum_normalized * AUXIN_NORMALIZATION_FACTOR;
    }
}

// Function to construct Von Neumann topology
void construct_von_neumann_topology(int current_pop_size, int *topology, int topology_size) {
    int rows = (int)sqrt((double)current_pop_size);
    int cols = (current_pop_size + rows - 1) / rows;
    for (int i = 0; i < topology_size; i++) {
        topology[i] = -1;
    }

    for (int i = 0; i < current_pop_size; i++) {
        int row = i / cols;
        int col = i % cols;
        if (col > 0) topology[i * 4 + 0] = i - 1;  // Left
        if (col < cols - 1 && i + 1 < current_pop_size) topology[i * 4 + 1] = i + 1;  // Right
        if (row > 0) topology[i * 4 + 2] = i - cols;  // Up
        if (row < (current_pop_size + cols - 1) / cols - 1 && i + cols < current_pop_size) topology[i * 4 + 3] = i + cols;  // Down
    }
}

// Main Root Regrowth Phase
void regrowth_phase(Optimizer *opt, int *topology, double *auxin, double *fitness, double *auxin_sorted) {
    calculate_auxin_concentration(opt, fitness, auxin);

    // Compute median auxin (sort once, reuse for lateral_growth_phase)
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles_arfo);
    double median_auxin = auxin_sorted[opt->population_size / 2];

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > median_auxin) {
            int valid_neighbors[4];
            int valid_count = 0;
            for (int k = 0; k < 4; k++) {
                if (topology[i * 4 + k] >= 0) {
                    valid_neighbors[valid_count++] = topology[i * 4 + k];
                }
            }

            int local_best_idx = i;
            if (valid_count > 0) {
                double min_fitness = fitness[valid_neighbors[0]];
                local_best_idx = valid_neighbors[0];
                for (int k = 1; k < valid_count; k++) {
                    if (fitness[valid_neighbors[k]] < min_fitness) {
                        min_fitness = fitness[valid_neighbors[k]];
                        local_best_idx = valid_neighbors[k];
                    }
                }
            }

            for (int j = 0; j < opt->dim; j++) {
                double rand_coeff = rand_double(0.0, 1.0);
                opt->population[i].position[j] += LOCAL_INERTIA * rand_coeff * 
                                                 (opt->population[local_best_idx].position[j] - opt->population[i].position[j]);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Root Branching Phase
void branching_phase(Optimizer *opt, int t, double *auxin, double *fitness, double *new_roots, int *new_root_count, FitnessIndex *fitness_indices) {
    calculate_auxin_concentration(opt, fitness, auxin);
    *new_root_count = 0;

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > BRANCHING_THRESHOLD) {
            double R1 = rand_double(0.0, 1.0);
            int num_new_roots = (int)(R1 * auxin[i] * (MAX_BRANCHING - MIN_BRANCHING) + MIN_BRANCHING);
            double std = ((opt->max_iter - t) / (double)opt->max_iter) * (INITIAL_STD - FINAL_STD) + FINAL_STD;

            for (int k = 0; k < num_new_roots && *new_root_count < opt->population_size * MAX_BRANCHING; k++) {
                for (int j = 0; j < opt->dim; j++) {
                    new_roots[*new_root_count * opt->dim + j] = opt->population[i].position[j] + std * rand_double(-1.0, 1.0);
                    new_roots[*new_root_count * opt->dim + j] = fmax(opt->bounds[2 * j], 
                                                                    fmin(opt->bounds[2 * j + 1], 
                                                                         new_roots[*new_root_count * opt->dim + j]));
                }
                (*new_root_count)++;
            }
        }
    }

    // Replace worst roots with new ones
    if (*new_root_count > 0) {
        // Update fitness_indices
        for (int i = 0; i < opt->population_size; i++) {
            fitness_indices[i].fitness = fitness[i];
            fitness_indices[i].index = i;
        }
        qsort(fitness_indices, opt->population_size, sizeof(FitnessIndex), compare_fitness_index);

        int replace_count = *new_root_count < opt->population_size ? *new_root_count : opt->population_size;
        for (int i = 0; i < replace_count; i++) {
            int worst_idx = fitness_indices[opt->population_size - 1 - i].index;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[worst_idx].position[j] = new_roots[i * opt->dim + j];
            }
        }
    }
}

// Lateral Root Growth Phase
void lateral_growth_phase(Optimizer *opt, double *auxin, double *fitness, double *auxin_sorted, double *new_roots) {
    // Reuse auxin_sorted from regrowth_phase for median
    double median_auxin = auxin_sorted[opt->population_size / 2];

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] <= median_auxin) {
            double rand_length = rand_double(0.0, 1.0) * MAX_ELONGATION;
            double norm = 0.0;
            double *random_vector = new_roots + i * opt->dim;  // Reuse new_roots buffer
            for (int j = 0; j < opt->dim; j++) {
                random_vector[j] = rand_double(-1.0, 1.0);
                norm += random_vector[j] * random_vector[j];
            }
            norm = sqrt(norm);

            for (int j = 0; j < opt->dim; j++) {
                double growth_angle = random_vector[j] / norm;
                opt->population[i].position[j] += rand_length * growth_angle;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Dead Root Elimination Phase
void elimination_phase_arfo(Optimizer *opt, double *auxin, double *fitness, double *auxin_sorted) {
    calculate_auxin_concentration(opt, fitness, auxin);
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles_arfo);
    double elimination_threshold = auxin_sorted[(int)(ELIMINATION_PERCENTILE * opt->population_size / 100)];

    int new_pop_size = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > elimination_threshold) {
            if (new_pop_size != i) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[new_pop_size].position[j] = opt->population[i].position[j];
                }
                opt->population[new_pop_size].fitness = fitness[i];
            }
            new_pop_size++;
        }
    }
    opt->population_size = new_pop_size;
}

// Population Replenishment Phase
void replenish_phase(Optimizer *opt, double *fitness, double (*objective_function)(double *)) {
    int target_pop_size = opt->population_size;  // Original population size stored elsewhere
    while (opt->population_size < target_pop_size) {
        int i = opt->population_size;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        fitness[i] = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness[i];
        opt->population_size++;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void ARFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Pre-allocate reusable arrays
    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *auxin_sorted = (double *)malloc(opt->population_size * sizeof(double));
    int *topology = (int *)malloc(opt->population_size * 4 * sizeof(int));
    double *new_roots = (double *)malloc(opt->population_size * MAX_BRANCHING * opt->dim * sizeof(double));
    FitnessIndex *fitness_indices = (FitnessIndex *)malloc(opt->population_size * sizeof(FitnessIndex));

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        fitness[i] = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness[i];
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[i];
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
        fitness_indices[i].fitness = fitness[i];
        fitness_indices[i].index = i;
    }
    qsort(fitness_indices, opt->population_size, sizeof(FitnessIndex), compare_fitness_index);

    int original_pop_size = opt->population_size;
    int new_root_count;

    // Construct initial topology
    construct_von_neumann_topology(opt->population_size, topology, opt->population_size * 4);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        regrowth_phase(opt, topology, auxin, fitness, auxin_sorted);
        branching_phase(opt, iter, auxin, fitness, new_roots, &new_root_count, fitness_indices);
        lateral_growth_phase(opt, auxin, fitness, auxin_sorted, new_roots);
        elimination_phase_arfo(opt, auxin, fitness, auxin_sorted);
        replenish_phase(opt, fitness, objective_function);

        // Update fitness and best solution
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness[i];
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update topology if population size changed
        construct_von_neumann_topology(opt->population_size, topology, opt->population_size * 4);

        // Reset population size to original
        opt->population_size = original_pop_size;
    }

    // Free allocated memory
    free(auxin);
    free(fitness);
    free(auxin_sorted);
    free(topology);
    free(new_roots);
    free(fitness_indices);
}
