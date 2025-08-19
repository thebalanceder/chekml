#include "VNS.h"
#include <time.h>

// Generate a random double between min and max
double rand_double_vns(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Generate a neighbor by applying scaled mutation within bounds
void generate_neighbor_vns(double *current_solution, double *neighbor, int dim, 
                           const double *bounds, int neighborhood_size) {
    for (int j = 0; j < dim; j++) {
        double range = bounds[2 * j + 1] - bounds[2 * j];
        double mutation = MUTATION_RATE * range * rand_double_vns(-0.5, 0.5) * neighborhood_size;
        neighbor[j] = current_solution[j] + mutation;

        // Clamp to bounds
        if (neighbor[j] < bounds[2 * j]) {
            neighbor[j] = bounds[2 * j];
        } else if (neighbor[j] > bounds[2 * j + 1]) {
            neighbor[j] = bounds[2 * j + 1];
        }
    }
}

// Variable Neighborhood Search Optimization (VND-style inside VNS)
void VNS_optimize(Optimizer *opt, double (*objective_function)(double *), 
                  int max_iterations, const int *neighborhood_sizes, int num_neighborhoods) {
    if (!opt || !objective_function || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "VNS_optimize: Null pointer input.\n");
        return;
    }

    int dim = opt->dim;
    const double *bounds = opt->bounds;

    double *current_solution = (double *)malloc(dim * sizeof(double));
    double *best_solution = (double *)malloc(dim * sizeof(double));
    double *neighbor_solution = (double *)malloc(dim * sizeof(double));

    if (!current_solution || !best_solution || !neighbor_solution) {
        fprintf(stderr, "VNS_optimize: Memory allocation failed.\n");
        free(current_solution);
        free(best_solution);
        free(neighbor_solution);
        return;
    }

    // Seed RNG (optional: make this configurable for reproducibility)
    srand(time(NULL));

    // Initialize random solution
    for (int j = 0; j < dim; j++) {
        current_solution[j] = rand_double_vns(bounds[2 * j], bounds[2 * j + 1]);
        best_solution[j] = current_solution[j];
    }

    double current_value = objective_function(current_solution);
    double best_value = current_value;

    // Main VNS loop
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int k = 0; k < num_neighborhoods; k++) {
            int neighborhood_size = neighborhood_sizes[k];

            generate_neighbor_vns(current_solution, neighbor_solution, dim, bounds, neighborhood_size);
            double neighbor_value = objective_function(neighbor_solution);

            if (neighbor_value < current_value) {
                // Accept improvement
                for (int j = 0; j < dim; j++) {
                    current_solution[j] = neighbor_solution[j];
                }
                current_value = neighbor_value;

                // Update global best
                if (current_value < best_value) {
                    for (int j = 0; j < dim; j++) {
                        best_solution[j] = current_solution[j];
                    }
                    best_value = current_value;
                }

                // Restart neighborhood search
                k = -1;
            }
        }

        // Optional: log progress
        // printf("Iter %d: Best fitness = %.10f\n", iter, best_value);
    }

    // Copy final best solution to optimizer struct
    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = best_solution[j];
    }
    opt->best_solution.fitness = best_value;

    // Cleanup
    free(current_solution);
    free(best_solution);
    free(neighbor_solution);
}

