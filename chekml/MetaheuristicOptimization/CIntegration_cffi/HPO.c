#include "HPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Quicksort implementation for sorting distances
void quicksort_indices(double *arr, int *indices, int low, int high) {
    if (low < high) {
        // Partition
        double pivot = arr[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[indices[j]] <= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        int pi = i + 1;

        // Recurse
        quicksort_indices(arr, indices, low, pi - 1);
        quicksort_indices(arr, indices, pi + 1, high);
    }
}

// Update positions for all individuals (Safe and Attack modes)
void hpo_update_positions(Optimizer *opt, int iter, double c_factor, double *xi, double *dist, int *idxsortdist, double *r1, double *r3, char *idx, double *z) {
    double c = 1.0 - ((double)iter * c_factor);
    int kbest = (int)(opt->population_size * c + 0.5);

    // Compute mean position (xi)
    for (int j = 0; j < opt->dim; j++) {
        xi[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            xi[j] += opt->population[i].position[j];
        }
        xi[j] /= opt->population_size;
    }

    // Compute distances to mean for sorting
    for (int i = 0; i < opt->population_size; i++) {
        dist[i] = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = xi[j] - opt->population[i].position[j];
            dist[i] += diff * diff;
        }
        dist[i] = sqrt(dist[i]);
        idxsortdist[i] = i;
    }

    // Sort distances using quicksort
    quicksort_indices(dist, idxsortdist, 0, opt->population_size - 1);

    // Update each individual's position
    for (int i = 0; i < opt->population_size; i++) {
        double r2 = rand_double(0.0, 1.0);

        // Generate r1 (boolean array based on c)
        for (int j = 0; j < opt->dim; j++) {
            r1[j] = rand_double(0.0, 1.0) < c ? 1.0 : 0.0;
            idx[j] = (r1[j] == 0.0);
            r3[j] = rand_double(0.0, 1.0);
            z[j] = idx[j] ? r2 : r3[j];
        }

        if (rand_double(0.0, 1.0) < CONSTRICTION_COEFF) {
            // Safe mode: Move towards mean and kbest-th closest individual
            int si_idx = idxsortdist[kbest - 1];
            for (int j = 0; j < opt->dim; j++) {
                double si_pos = opt->population[si_idx].position[j];
                opt->population[i].position[j] += 0.5 * (
                    (2.0 * c * z[j] * si_pos - opt->population[i].position[j]) +
                    (2.0 * (1.0 - c) * z[j] * xi[j] - opt->population[i].position[j])
                );
            }
        } else {
            // Attack mode: Move towards target with cosine perturbation
            for (int j = 0; j < opt->dim; j++) {
                double rr = -1.0 + 2.0 * z[j];
                opt->population[i].position[j] = 2.0 * z[j] * cos(TWO_PI * rr) * 
                    (opt->best_solution.position[j] - opt->population[i].position[j]) + 
                    opt->best_solution.position[j];
            }
        }

        // Enforce bounds
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Main Optimization Function
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate temporary data
    double c_factor = C_PARAM_MAX / opt->max_iter;
    double *xi = (double *)calloc(opt->dim, sizeof(double));
    double *dist = (double *)calloc(opt->population_size, sizeof(double));
    int *idxsortdist = (int *)calloc(opt->population_size, sizeof(int));
    double *r1 = (double *)calloc(opt->dim, sizeof(double));
    double *r3 = (double *)calloc(opt->dim, sizeof(double));
    char *idx = (char *)calloc(opt->dim, sizeof(char));
    double *z = (double *)calloc(opt->dim, sizeof(double));

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

    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        hpo_update_positions(opt, iter, c_factor, xi, dist, idxsortdist, r1, r3, idx, z);

        // Evaluate new positions and update best solution
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

        printf("Iteration: %d, Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(xi);
    free(dist);
    free(idxsortdist);
    free(r1);
    free(r3);
    free(idx);
    free(z);
}
