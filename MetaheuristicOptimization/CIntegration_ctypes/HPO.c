#include "HPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Update positions for all individuals (Safe and Attack modes)
void hpo_update_positions(Optimizer *opt, int iter) {
    double c = 1.0 - ((double)iter * C_PARAM_MAX / opt->max_iter);
    int kbest = (int)(opt->population_size * c + 0.5); // Round to nearest integer
    double *xi = (double *)calloc(opt->dim, sizeof(double));
    double *dist = (double *)calloc(opt->population_size, sizeof(double));
    int *idxsortdist = (int *)calloc(opt->population_size, sizeof(int));

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
            dist[i] += (xi[j] - opt->population[i].position[j]) * (xi[j] - opt->population[i].position[j]);
        }
        dist[i] = sqrt(dist[i]);
        idxsortdist[i] = i;
    }

    // Sort distances (simple bubble sort for simplicity)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (dist[idxsortdist[j]] > dist[idxsortdist[j + 1]]) {
                int temp = idxsortdist[j];
                idxsortdist[j] = idxsortdist[j + 1];
                idxsortdist[j + 1] = temp;
            }
        }
    }

    // Update each individual's position
    for (int i = 0; i < opt->population_size; i++) {
        double r2 = rand_double(0.0, 1.0);
        double *r1 = (double *)calloc(opt->dim, sizeof(double));
        double *r3 = (double *)calloc(opt->dim, sizeof(double));
        char *idx = (char *)calloc(opt->dim, sizeof(char));
        double *z = (double *)calloc(opt->dim, sizeof(double));

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
                opt->population[i].position[j] = 2.0 * z[j] * cos(2.0 * M_PI * rr) * 
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

        free(r1);
        free(r3);
        free(idx);
        free(z);
    }

    free(xi);
    free(dist);
    free(idxsortdist);
}

// Main Optimization Function
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
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
        hpo_update_positions(opt, iter);

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
}
