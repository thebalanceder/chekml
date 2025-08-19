#include "VNS.h"
#include <time.h>

// Generate a random double between min and max
inline double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Generate neighbor with mutation, using `restrict` for faster memory access
void generate_neighbor_vns(const double * restrict current, double * restrict neighbor, int dim, 
                           const double * restrict bounds, int neighborhood_size) {
    for (int j = 0; j < dim; ++j) {
        const double lower = bounds[2 * j];
        const double upper = bounds[2 * j + 1];
        const double range = upper - lower;

        double mutation = MUTATION_RATE * range * rand_double(-0.5, 0.5) * neighborhood_size;
        double value = current[j] + mutation;

        // Branchless clamp
        value = value < lower ? lower : (value > upper ? upper : value);
        neighbor[j] = value;
    }
}

// Optimized VNS
void VNS_optimize(Optimizer *opt, double (*objective_function)(double *), 
                  int max_iterations, const int *neighborhood_sizes, int num_neighborhoods) {
    if (!opt || !objective_function || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "VNS_optimize: Null pointer input.\n");
        return;
    }

    const int dim = opt->dim;
    const double * restrict bounds = opt->bounds;
    double * restrict current = (double *)malloc(dim * sizeof(double));
    double * restrict best = (double *)malloc(dim * sizeof(double));
    double * restrict neighbor = (double *)malloc(dim * sizeof(double));

    if (!current || !best || !neighbor) {
        fprintf(stderr, "VNS_optimize: Memory allocation failed.\n");
        free(current); free(best); free(neighbor);
        return;
    }

    srand((unsigned int)time(NULL));  // RNG seed

    // Init random solution
    for (int j = 0; j < dim; ++j) {
        double val = rand_double(bounds[2 * j], bounds[2 * j + 1]);
        current[j] = val;
        best[j] = val;
    }

    double current_value = objective_function(current);
    double best_value = current_value;

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int k = 0; k < num_neighborhoods; ++k) {
            int neighborhood_size = neighborhood_sizes[k];

            generate_neighbor_vns(current, neighbor, dim, bounds, neighborhood_size);
            double neighbor_value = objective_function(neighbor);

            if (neighbor_value < current_value) {
                // Swap pointers for current/neighbor
                double *temp = current;
                current = neighbor;
                neighbor = temp;
                current_value = neighbor_value;

                if (current_value < best_value) {
                    for (int j = 0; j < dim; ++j) best[j] = current[j];
                    best_value = current_value;
                }

                k = -1; // Restart neighborhood search
            }
        }
    }

    // Store best solution in optimizer
    for (int j = 0; j < dim; ++j) {
        opt->best_solution.position[j] = best[j];
    }
    opt->best_solution.fitness = best_value;

    // Clean up
    free(current);
    free(best);
    free(neighbor);
}

