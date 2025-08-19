#include "VNS.h"
#include <time.h>
#include <math.h>
#include <float.h>

// Fast random double generator
static inline double rand_double_vns(double min, double max) {
    return min + ((double)rand() / (double)RAND_MAX) * (max - min);
}

// Branchless clamp using fmin/fmax
static inline double clamp(double x, double low, double high) {
    return fmax(low, fmin(x, high));
}

// Generate a neighbor
void generate_neighbor_vns(const double *restrict current_solution,
                           double *restrict neighbor,
                           int dim,
                           const double *restrict bounds,
                           int neighborhood_size) {
    for (int j = 0; j < dim; ++j) {
        double range = bounds[2 * j + 1] - bounds[2 * j];
        double mutation = MUTATION_RATE * range * rand_double_vns(-0.5, 0.5) * neighborhood_size;
        neighbor[j] = clamp(current_solution[j] + mutation, bounds[2 * j], bounds[2 * j + 1]);
    }
}

// Variable Neighborhood Search Optimization
void VNS_optimize(Optimizer *opt, double (*objective_function)(double *),
                  int max_iterations, const int *neighborhood_sizes, int num_neighborhoods) {
    if (!opt || !objective_function || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "VNS_optimize: Null pointer input.\n");
        return;
    }

    const int dim = opt->dim;
    const double *restrict bounds = opt->bounds;

    double *restrict current_solution = (double *)malloc(dim * sizeof(double));
    double *restrict best_solution = (double *)malloc(dim * sizeof(double));
    double *restrict neighbor_solution = (double *)malloc(dim * sizeof(double));

    if (!current_solution || !best_solution || !neighbor_solution) {
        fprintf(stderr, "VNS_optimize: Memory allocation failed.\n");
        free(current_solution);
        free(best_solution);
        free(neighbor_solution);
        return;
    }

    // Seed RNG once externally or here (optional)
    srand((unsigned int)time(NULL));

    // Random initialization
    for (int j = 0; j < dim; ++j) {
        current_solution[j] = rand_double_vns(bounds[2 * j], bounds[2 * j + 1]);
        best_solution[j] = current_solution[j];
    }

    double current_value = objective_function(current_solution);
    double best_value = current_value;

    // VNS loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int k = 0; k < num_neighborhoods; ++k) {
            int neighborhood_size = neighborhood_sizes[k];

            generate_neighbor_vns(current_solution, neighbor_solution, dim, bounds, neighborhood_size);
            double neighbor_value = objective_function(neighbor_solution);

            if (neighbor_value < current_value) {
                // Accept neighbor
                for (int j = 0; j < dim; ++j) {
                    current_solution[j] = neighbor_solution[j];
                }
                current_value = neighbor_value;

                // Update best if improved
                if (current_value < best_value) {
                    for (int j = 0; j < dim; ++j) {
                        best_solution[j] = current_solution[j];
                    }
                    best_value = current_value;
                }

                k = -1; // Reset neighborhood search
            }
        }
    }

    // Copy final best to output
    for (int j = 0; j < dim; ++j) {
        opt->best_solution.position[j] = best_solution[j];
    }
    opt->best_solution.fitness = best_value;

    free(current_solution);
    free(best_solution);
    free(neighbor_solution);
}

