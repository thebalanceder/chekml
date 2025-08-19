#include "DEA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Random double generator
double rand_double(double min, double max);

// Initialize dolphin locations and alternatives
void initialize_locations(Optimizer *opt, DEAData *dea_data) {
    dea_data->dim_data = (DimensionData *)malloc(opt->dim * sizeof(DimensionData));
    
    // Calculate effective radius
    double min_search_size = INFINITY;
    for (int j = 0; j < opt->dim; j++) {
        double size = fabs(opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        if (size < min_search_size) min_search_size = size;
    }
    dea_data->effective_radius = DEA_EFFECTIVE_RADIUS_FACTOR * min_search_size;

    // Initialize alternatives per dimension
    for (int j = 0; j < opt->dim; j++) {
        DimensionData *dim = &dea_data->dim_data[j];
        dim->size = DEA_ALTERNATIVES_PER_DIM;
        dim->values = (double *)malloc(dim->size * sizeof(double));
        dim->accum_fitness = (double *)calloc(dim->size, sizeof(double));
        dim->probabilities = (double *)calloc(dim->size, sizeof(double));
        
        // Linearly space alternatives
        double lower = opt->bounds[2 * j];
        double upper = opt->bounds[2 * j + 1];
        for (int k = 0; k < dim->size; k++) {
            dim->values[k] = lower + (upper - lower) * k / (dim->size - 1);
        }
    }

    // Initialize random population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Calculate accumulative fitness for alternatives
void calculate_accumulative_fitness(Optimizer *opt, double *fitness, DEAData *dea_data) {
    // Reset accumulative fitness
    for (int j = 0; j < opt->dim; j++) {
        memset(dea_data->dim_data[j].accum_fitness, 0, dea_data->dim_data[j].size * sizeof(double));
    }

    // Distribute fitness to alternatives and neighbors
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            DimensionData *dim = &dea_data->dim_data[j];
            double loc_value = opt->population[i].position[j];
            
            // Find closest alternative
            int alt_idx = 0;
            double min_diff = fabs(dim->values[0] - loc_value);
            for (int k = 1; k < dim->size; k++) {
                double diff = fabs(dim->values[k] - loc_value);
                if (diff < min_diff) {
                    min_diff = diff;
                    alt_idx = k;
                }
            }

            // Distribute fitness to neighbors within effective radius
            for (int k = -10; k <= 10; k++) {
                int neighbor_idx = alt_idx + k;
                if (neighbor_idx >= 0 && neighbor_idx < dim->size) {
                    double weight = (10.0 - fabs((double)k)) / 10.0;
                    dim->accum_fitness[neighbor_idx] += weight * fitness[i];
                }
            }
        }
    }
}

// Compute convergence probability based on loop
double get_convergence_probability(int loop, int max_loops) {
    double t = (double)loop / max_loops;
    return pow(t, DEA_CONVERGENCE_POWER);
}

// Update probabilities for alternatives
void update_probabilities(Optimizer *opt, int loop, double *fitness, DEAData *dea_data) {
    double convergence_prob = get_convergence_probability(loop, opt->max_iter);

    // Find best alternative indices
    int *best_alt_indices = (int *)malloc(opt->dim * sizeof(int));
    for (int j = 0; j < opt->dim; j++) {
        DimensionData *dim = &dea_data->dim_data[j];
        double loc_value = opt->best_solution.position[j];
        int alt_idx = 0;
        double min_diff = fabs(dim->values[0] - loc_value);
        for (int k = 1; k < dim->size; k++) {
            double diff = fabs(dim->values[k] - loc_value);
            if (diff < min_diff) {
                min_diff = diff;
                alt_idx = k;
            }
        }
        best_alt_indices[j] = alt_idx;
    }

    // Assign probabilities
    for (int j = 0; j < opt->dim; j++) {
        DimensionData *dim = &dea_data->dim_data[j];
        double total_af = 0.0;
        for (int k = 0; k < dim->size; k++) {
            total_af += dim->accum_fitness[k];
        }

        if (total_af == 0.0) {
            for (int k = 0; k < dim->size; k++) {
                dim->probabilities[k] = 1.0 / dim->size;
            }
        } else {
            // Best alternative gets convergence probability
            dim->probabilities[best_alt_indices[j]] = convergence_prob;
            double remaining_prob = 1.0 - convergence_prob;

            // Distribute remaining probability
            for (int k = 0; k < dim->size; k++) {
                if (k != best_alt_indices[j]) {
                    double prob = (dim->accum_fitness[k] / total_af) * remaining_prob;
                    dim->probabilities[k] = fmax(prob, DEA_PROBABILITY_THRESHOLD);
                }
            }

            // Normalize probabilities
            double prob_sum = 0.0;
            for (int k = 0; k < dim->size; k++) {
                prob_sum += dim->probabilities[k];
            }
            if (prob_sum > 0.0) {
                for (int k = 0; k < dim->size; k++) {
                    dim->probabilities[k] /= prob_sum;
                }
            }
        }
    }
    free(best_alt_indices);
}

// Select new locations based on probabilities
void select_new_locations(Optimizer *opt, DEAData *dea_data) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            DimensionData *dim = &dea_data->dim_data[j];
            double r = rand_double(0.0, 1.0);
            double cum_prob = 0.0;
            int selected_idx = 0;

            // Select alternative based on cumulative probability
            for (int k = 0; k < dim->size; k++) {
                cum_prob += dim->probabilities[k];
                if (r <= cum_prob) {
                    selected_idx = k;
                    break;
                }
            }
            opt->population[i].position[j] = dim->values[selected_idx];
        }
    }
    enforce_bound_constraints(opt);
}

// Main DEA optimization function
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double prev_best_fitness = INFINITY;

    // Initialize DEA data
    DEAData *dea_data = (DEAData *)malloc(sizeof(DEAData));
    initialize_locations(opt, dea_data);
    
    for (int loop = 0; loop < opt->max_iter; loop++) {
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update probabilities and select new locations
        calculate_accumulative_fitness(opt, fitness, dea_data);
        update_probabilities(opt, loop, fitness, dea_data);
        select_new_locations(opt, dea_data);

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
        free(dea_data->dim_data[j].values);
        free(dea_data->dim_data[j].accum_fitness);
        free(dea_data->dim_data[j].probabilities);
    }
    free(dea_data->dim_data);
    free(dea_data);
    free(fitness);
}
