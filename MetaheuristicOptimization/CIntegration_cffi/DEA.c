#include "DEA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Precomputed weights for fitness distribution (-10 to 10)
static const double WEIGHTS[21] = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0
};

// Random double generator (batch generation could be added if needed)
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize dolphin locations and alternatives
void initialize_locations(Optimizer *opt, DEAData *dea_data) {
    const int dim = opt->dim;
    dea_data->dim_data = (DimensionData *)malloc(dim * sizeof(DimensionData));
    
    // Calculate effective radius
    double min_search_size = INFINITY;
    for (int j = 0; j < dim; j++) {
        double size = fabs(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        if (size < min_search_size) min_search_size = size;
    }
    dea_data->effective_radius = DEA_EFFECTIVE_RADIUS_FACTOR * min_search_size;

    // Initialize alternatives per dimension
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        dim->size = DEA_ALTERNATIVES_PER_DIM;
        dim->values = (double *)malloc(dim->size * sizeof(double));
        dim->accum_fitness = (double *)malloc(dim->size * sizeof(double));
        dim->probabilities = (double *)malloc(dim->size * sizeof(double));
        
        // Linearly space alternatives
        double lower = opt->bounds[2 * j];
        double upper = opt->bounds[2 * j + 1];
        double step = (upper - lower) / (dim->size - 1);
        double *values = dim->values;
        for (int k = 0; k < dim->size; k++) {
            values[k] = lower + k * step;
            dim->accum_fitness[k] = 0.0;
            dim->probabilities[k] = 0.0;
        }
    }

    // Initialize random population
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Binary search to find closest alternative index
static inline int find_closest_alternative(const double *values, int size, double target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (values[mid] == target) return mid;
        if (values[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    // Compare left-1 and left (or right and right+1)
    if (left == 0) return 0;
    if (left == size) return size - 1;
    return (fabs(values[left - 1] - target) < fabs(values[left] - target)) ? left - 1 : left;
}

// Calculate accumulative fitness for alternatives
void calculate_accumulative_fitness(Optimizer *opt, double *fitness, DEAData *dea_data) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    // Reset accumulative fitness
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        double *accum_fitness = dim->accum_fitness;
        for (int k = 0; k < dim->size; k++) {
            accum_fitness[k] = 0.0;
        }
    }

    // Distribute fitness to alternatives and neighbors
    for (int i = 0; i < pop_size; i++) {
        double fit = fitness[i];
        for (int j = 0; j < dim; j++) {
            DimensionData *dim = dea_data->dim_data + j;
            double loc_value = opt->population[i].position[j];
            
            // Find closest alternative using binary search
            int alt_idx = find_closest_alternative(dim->values, dim->size, loc_value);
            double *accum_fitness = dim->accum_fitness;

            // Distribute fitness to neighbors within effective radius
            for (int k = -10; k <= 10; k++) {
                int neighbor_idx = alt_idx + k;
                if (neighbor_idx >= 0 && neighbor_idx < dim->size) {
                    accum_fitness[neighbor_idx] += WEIGHTS[k + 10] * fit;
                }
            }
        }
    }
}

// Compute convergence probability based on loop
static inline double get_convergence_probability(int loop, int max_loops) {
    return pow((double)loop / max_loops, DEA_CONVERGENCE_POWER);
}

// Update probabilities for alternatives
void update_probabilities(Optimizer *opt, int loop, double *fitness, DEAData *dea_data) {
    const int dim = opt->dim;
    double convergence_prob = get_convergence_probability(loop, opt->max_iter);

    // Pre-allocate best alternative indices
    int *best_alt_indices = (int *)malloc(dim * sizeof(int));
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        best_alt_indices[j] = find_closest_alternative(dim->values, dim->size, opt->best_solution.position[j]);
    }

    // Assign probabilities
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        double *accum_fitness = dim->accum_fitness;
        double *probs = dim->probabilities;
        const int size = dim->size;
        double total_af = 0.0;

        for (int k = 0; k < size; k++) {
            total_af += accum_fitness[k];
        }

        if (total_af == 0.0) {
            double uniform_prob = 1.0 / size;
            for (int k = 0; k < size; k++) {
                probs[k] = uniform_prob;
            }
        } else {
            // Best alternative gets convergence probability
            int best_idx = best_alt_indices[j];
            probs[best_idx] = convergence_prob;
            double remaining_prob = 1.0 - convergence_prob;
            double prob_sum = convergence_prob;

            // Distribute and normalize in one pass
            for (int k = 0; k < size; k++) {
                if (k != best_idx) {
                    double prob = (accum_fitness[k] / total_af) * remaining_prob;
                    probs[k] = prob > DEA_PROBABILITY_THRESHOLD ? prob : DEA_PROBABILITY_THRESHOLD;
                    prob_sum += probs[k];
                }
            }

            // Normalize probabilities
            if (prob_sum > 0.0) {
                for (int k = 0; k < size; k++) {
                    probs[k] /= prob_sum;
                }
            }
        }
    }
    free(best_alt_indices);
}

// Select new locations based on probabilities
void select_new_locations(Optimizer *opt, DEAData *dea_data) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            DimensionData *dim = dea_data->dim_data + j;
            double *probs = dim->probabilities;
            double r = rand_double(0.0, 1.0);
            double cum_prob = 0.0;
            int selected_idx = 0;

            // Select alternative based on cumulative probability
            for (int k = 0; k < dim->size; k++) {
                cum_prob += probs[k];
                if (r <= cum_prob) {
                    selected_idx = k;
                    break;
                }
            }
            position[j] = dim->values[selected_idx];
        }
    }
    enforce_bound_constraints(opt);
}

// Main DEA optimization function
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    double prev_best_fitness = INFINITY;

    // Initialize DEA data
    DEAData dea_data;
    initialize_locations(opt, &dea_data);
    
    for (int loop = 0; loop < opt->max_iter; loop++) {
        // Evaluate fitness
        for (int i = 0; i < pop_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Update probabilities and select new locations
        calculate_accumulative_fitness(opt, fitness, &dea_data);
        update_probabilities(opt, loop, fitness, &dea_data);
        select_new_locations(opt, &dea_data);

        // Check convergence
        if (loop > 0 && fabs(opt->best_solution.fitness - prev_best_fitness) < 1e-6) {
            printf("Convergence reached at loop %d.\n", loop + 1);
            break;
        }
        prev_best_fitness = opt->best_solution.fitness;
        printf("Loop %d: Best Value = %f\n", loop + 1, opt->best_solution.fitness);
    }

    // Cleanup
    for (int j = 0; j < opt->dim; j++) {
        free(dea_data.dim_data[j].values);
        free(dea_data.dim_data[j].accum_fitness);
        free(dea_data.dim_data[j].probabilities);
    }
    free(dea_data.dim_data);
    free(fitness);
}
