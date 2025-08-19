/* ARFO.c - Implementation file for Artificial Root Foraging Optimization */
#include "ARFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Comparison function for qsort on doubles
int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
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
void regrowth_phase(Optimizer *opt) {
    int *topology = (int *)malloc(opt->population_size * 4 * sizeof(int));
    construct_von_neumann_topology(opt->population_size, topology, opt->population_size * 4);

    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
    }
    calculate_auxin_concentration(opt, fitness, auxin);

    double *auxin_sorted = (double *)malloc(opt->population_size * sizeof(double));
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles);
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

    free(topology);
    free(auxin);
    free(fitness);
    free(auxin_sorted);
}

// Main Root Branching Phase
void branching_phase(Optimizer *opt, int t) {
    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
    }
    calculate_auxin_concentration(opt, fitness, auxin);

    double *new_roots = (double *)malloc(opt->population_size * MAX_BRANCHING * opt->dim * sizeof(double));
    int new_root_count = 0;

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > BRANCHING_THRESHOLD) {
            double R1 = rand_double(0.0, 1.0);
            int num_new_roots = (int)(R1 * auxin[i] * (MAX_BRANCHING - MIN_BRANCHING) + MIN_BRANCHING);
            double std = ((opt->max_iter - t) / (double)opt->max_iter) * (INITIAL_STD - FINAL_STD) + FINAL_STD;

            for (int k = 0; k < num_new_roots; k++) {
                for (int j = 0; j < opt->dim; j++) {
                    new_roots[new_root_count * opt->dim + j] = opt->population[i].position[j] + std * rand_double(-1.0, 1.0);
                    new_roots[new_root_count * opt->dim + j] = fmax(opt->bounds[2 * j], 
                                                                    fmin(opt->bounds[2 * j + 1], 
                                                                         new_roots[new_root_count * opt->dim + j]));
                }
                new_root_count++;
            }
        }
    }

    // Replace worst roots with new ones if population is at capacity
    if (new_root_count > 0) {
        // Sort population by fitness to identify worst roots
        double *sorted_fitness = (double *)malloc(opt->population_size * sizeof(double));
        int *indices = (int *)malloc(opt->population_size * sizeof(int));
        for (int i = 0; i < opt->population_size; i++) {
            sorted_fitness[i] = opt->population[i].fitness;
            indices[i] = i;
        }
        // Bubble sort for simplicity (could use qsort for better performance)
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (sorted_fitness[j] > sorted_fitness[j + 1]) {
                    double temp_f = sorted_fitness[j];
                    sorted_fitness[j] = sorted_fitness[j + 1];
                    sorted_fitness[j + 1] = temp_f;
                    int temp_idx = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp_idx;
                }
            }
        }

        // Replace worst roots with new roots
        int replace_count = new_root_count < opt->population_size ? new_root_count : opt->population_size;
        for (int i = 0; i < replace_count; i++) {
            int worst_idx = indices[opt->population_size - 1 - i];
            for (int j = 0; j < opt->dim; j++) {
                opt->population[worst_idx].position[j] = new_roots[i * opt->dim + j];
            }
        }

        free(sorted_fitness);
        free(indices);
    }

    free(auxin);
    free(fitness);
    free(new_roots);
}

// Lateral Root Growth Phase
void lateral_growth_phase(Optimizer *opt) {
    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
    }
    calculate_auxin_concentration(opt, fitness, auxin);

    double *auxin_sorted = (double *)malloc(opt->population_size * sizeof(double));
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles);
    double median_auxin = auxin_sorted[opt->population_size / 2];

    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] <= median_auxin) {
            double rand_length = rand_double(0.0, 1.0) * MAX_ELONGATION;
            double *random_vector = (double *)malloc(opt->dim * sizeof(double));
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                random_vector[j] = rand_double(-1.0, 1.0);
                norm += random_vector[j] * random_vector[j];
            }
            norm = sqrt(norm);

            for (int j = 0; j < opt->dim; j++) {
                double growth_angle = random_vector[j] / norm;
                opt->population[i].position[j] += rand_length * growth_angle;
            }
            free(random_vector);
        }
    }
    enforce_bound_constraints(opt);

    free(auxin);
    free(fitness);
    free(auxin_sorted);
}

// Dead Root Elimination Phase
void elimination_phase_arfo(Optimizer *opt) {
    double *auxin = (double *)malloc(opt->population_size * sizeof(double));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
    }
    calculate_auxin_concentration(opt, fitness, auxin);

    double *auxin_sorted = (double *)malloc(opt->population_size * sizeof(double));
    memcpy(auxin_sorted, auxin, opt->population_size * sizeof(double));
    qsort(auxin_sorted, opt->population_size, sizeof(double), compare_doubles);
    double elimination_threshold = auxin_sorted[(int)(ELIMINATION_PERCENTILE * opt->population_size / 100)];

    int new_pop_size = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (auxin[i] > elimination_threshold) {
            if (new_pop_size != i) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[new_pop_size].position[j] = opt->population[i].position[j];
                }
                opt->population[new_pop_size].fitness = opt->population[i].fitness;
            }
            new_pop_size++;
        }
    }
    opt->population_size = new_pop_size;

    free(auxin);
    free(fitness);
    free(auxin_sorted);
}

// Population Replenishment Phase
void replenish_phase(Optimizer *opt) {
    while (opt->population_size < opt->population_size) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[opt->population_size].position[j] = opt->bounds[2 * j] + 
                                                               rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population_size++;
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void ARFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    int original_pop_size = opt->population_size;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        regrowth_phase(opt);
        branching_phase(opt, iter);
        lateral_growth_phase(opt);
        elimination_phase_arfo(opt);
        replenish_phase(opt);

        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);

        // Reset population size to original if needed
        opt->population_size = original_pop_size;
    }
}
