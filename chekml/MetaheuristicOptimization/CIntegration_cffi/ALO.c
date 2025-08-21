#include "ALO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>  // For time() function

// Quicksort implementation with indices
void quicksort_with_indices(double *arr, int *indices, int low, int high) {
    if (low < high) {
        double pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                int temp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_idx;
            }
        }

        double temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        int temp_idx = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp_idx;

        int pi = i + 1;
        quicksort_with_indices(arr, indices, low, pi - 1);
        quicksort_with_indices(arr, indices, pi + 1, high);
    }
}

// Initialize antlion and ant populations
void initialize_populations(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = rand_double_alo(lb, ub);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Roulette wheel selection based on fitness weights
int roulette_wheel_selection(double *weights, int size) {
    double accumulation = 0.0;
    for (int i = 0; i < size; i++) {
        accumulation += weights[i];
    }

    double p = rand_double_alo(0.0, accumulation);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += weights[i];
        if (cumsum > p) return i;
    }
    return 0;
}

// Random walk phase around a given antlion (restored full walk computation)
void random_walk_phase(Optimizer *opt, int t, double *antlion, double *walk_buffer) {
    double I = 1.0;
    double T = (double)opt->max_iter;

    // Adjust I based on iteration
    if (t > T / 10) I = 1.0 + I_FACTOR_1 * (t / T);
    if (t > T / 2) I = 1.0 + I_FACTOR_2 * (t / T);
    if (t > T * 3 / 4) I = 1.0 + I_FACTOR_3 * (t / T);
    if (t > T * 0.9) I = 1.0 + I_FACTOR_4 * (t / T);
    if (t > T * 0.95) I = 1.0 + I_FACTOR_5 * (t / T);

    // Compute bounds
    double lb[opt->dim], ub[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        lb[j] = opt->bounds[2 * j] / I;
        ub[j] = opt->bounds[2 * j + 1] / I;
    }

    // Move interval around antlion
    double r = rand_double_alo(0.0, 1.0);
    if (r < 0.5) {
        for (int j = 0; j < opt->dim; j++) lb[j] += antlion[j];
    } else {
        for (int j = 0; j < opt->dim; j++) lb[j] = -lb[j] + antlion[j];
    }

    r = rand_double_alo(0.0, 1.0);
    if (r >= 0.5) {
        for (int j = 0; j < opt->dim; j++) ub[j] += antlion[j];
    } else {
        for (int j = 0; j < opt->dim; j++) ub[j] = -ub[j] + antlion[j];
    }

    // Generate full random walk
    double *X = (double *)calloc(opt->max_iter + 1, sizeof(double));
    if (!X) return;

    for (int j = 0; j < opt->dim; j++) {
        // Reset X for each dimension
        memset(X, 0, (opt->max_iter + 1) * sizeof(double));
        for (int k = 0; k < opt->max_iter; k++) {
            r = rand_double_alo(0.0, 1.0);
            X[k + 1] = X[k] + (r > 0.5 ? 1.0 : -1.0);
        }

        double a = X[0], b = X[0];
        for (int k = 0; k <= opt->max_iter; k++) {
            if (X[k] < a) a = X[k];
            if (X[k] > b) b = X[k];
        }

        double c = lb[j], d = ub[j];
        walk_buffer[j] = ((X[t] - a) * (d - c)) / (b - a + 1e-10) + c;
    }

    free(X);
}

// Update ant positions based on random walks
void update_ant_positions(Optimizer *opt, int t, double *antlion_positions, double *walk_buffer) {
    double weights[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        weights[i] = 1.0 / (opt->population[i].fitness + ROULETTE_EPSILON);
    }

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;

        // Select antlion using roulette wheel
        int roulette_idx = roulette_wheel_selection(weights, opt->population_size);
        if (roulette_idx == -1) roulette_idx = 0;  // Fallback

        // Random walk around selected antlion
        random_walk_phase(opt, t, &antlion_positions[roulette_idx * opt->dim], walk_buffer);
        double RA[opt->dim];
        for (int j = 0; j < opt->dim; j++) RA[j] = walk_buffer[j];

        // Random walk around elite antlion
        random_walk_phase(opt, t, opt->best_solution.position, walk_buffer);
        double RE[opt->dim];
        for (int j = 0; j < opt->dim; j++) RE[j] = walk_buffer[j];

        // Update ant position (Equation 2.13)
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = (RA[j] + RE[j]) / 2.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Update antlions by combining and sorting populations
void update_antlions_phase(Optimizer *opt, double *antlion_positions) {
    int total_size = 2 * opt->population_size;
    double *combined_fitness = (double *)malloc(total_size * sizeof(double));
    int *indices = (int *)malloc(total_size * sizeof(int));
    if (!combined_fitness || !indices) {
        free(combined_fitness); free(indices);
        return;
    }

    // Copy antlion fitness
    for (int i = 0; i < opt->population_size; i++) {
        combined_fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }

    // Copy ant fitness
    for (int i = 0; i < opt->population_size; i++) {
        combined_fitness[opt->population_size + i] = opt->population[i].fitness;
        indices[opt->population_size + i] = opt->population_size + i;
    }

    // Sort using quicksort
    quicksort_with_indices(combined_fitness, indices, 0, total_size - 1);

    // Update antlion positions
    for (int i = 0; i < opt->population_size; i++) {
        int idx = indices[i];
        double *src = (idx < opt->population_size) ? &antlion_positions[idx * opt->dim] : opt->population[idx - opt->population_size].position;
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = src[j];
        }
        opt->population[i].fitness = combined_fitness[i];
    }

    // Update elite if better solution found
    if (combined_fitness[0] < opt->best_solution.fitness) {
        opt->best_solution.fitness = combined_fitness[0];
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = antlion_positions[j];
        }
    }

    // Ensure elite is in population
    for (int j = 0; j < opt->dim; j++) {
        antlion_positions[j] = opt->best_solution.position[j];
    }
    opt->population[0].fitness = opt->best_solution.fitness;

    free(combined_fitness);
    free(indices);
}

// Main Optimization Function
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Allocate antlion positions and walk buffer
    double *antlion_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *walk_buffer = (double *)malloc(opt->dim * sizeof(double));
    if (!antlion_positions || !walk_buffer) {
        free(antlion_positions); free(walk_buffer);
        return;
    }

    // Initialize populations
    initialize_populations(opt);
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = opt->population[i].position[j];
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        // Debug: Print initial fitness
        printf("Initial antlion %d fitness: %f\n", i, opt->population[i].fitness);
    }

    // Set initial elite
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
    // Debug: Print initial elite
    printf("Initial elite solution: [%f, %f], Fitness: %f\n",
           opt->best_solution.position[0], opt->best_solution.position[1],
           opt->best_solution.fitness);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        update_ant_positions(opt, t, antlion_positions, walk_buffer);
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        update_antlions_phase(opt, antlion_positions);

        // Debug: Print elite solution every iteration
        printf("Iteration %d, Elite solution: [%f, %f], Fitness: %f\n",
               t + 1, opt->best_solution.position[0], opt->best_solution.position[1],
               opt->best_solution.fitness);

        // Log progress every 50 iterations
        if ((t + 1) % 50 == 0) {
            printf("At iteration %d, the elite fitness is %f\n", t + 1, opt->best_solution.fitness);
        }
    }

    free(antlion_positions);
    free(walk_buffer);
}
