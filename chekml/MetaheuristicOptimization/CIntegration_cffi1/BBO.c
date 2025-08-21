#include "BBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

#ifndef NDEBUG
#include <stdio.h>
#endif

// Fast linear congruential generator (LCG)
static unsigned long lcg_state = 1;
static inline unsigned long lcg_rand() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state;
}

static inline double lcg_rand_double(double min, double max) {
    return min + (max - min) * ((double)(lcg_rand() >> 11) / (double)(1ULL << 53));
}

// Generate Gaussian random number using Box-Muller transform
static inline void generate_normal_pair(double *n1, double *n2) {
    double u1 = lcg_rand_double(0.0, 1.0);
    double u2 = lcg_rand_double(0.0, 1.0);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *n1 = r * cos(theta);
    *n2 = r * sin(theta);
}

// Compute migration rates (mu and lambda)
static inline void compute_migration_rates(double *mu, double *lambda_, int population_size) {
    for (int i = 0; i < population_size; i++) {
        mu[i] = 1.0 - ((double)i / (population_size - 1));
        lambda_[i] = 1.0 - mu[i];
    }
}

// Roulette wheel selection
static inline int roulette_wheel_selection(double *probabilities, int size) {
    double r = lcg_rand_double(0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += probabilities[i];
        if (r <= cumsum) return i;
    }
    return size - 1;
}

// Custom quicksort for indices
static void quicksort_indices(double *fitness, int *indices, int left, int right) {
    if (left >= right) return;
    
    double pivot = fitness[indices[(left + right) / 2]];
    int i = left, j = right;
    
    while (i <= j) {
        while (fitness[indices[i]] < pivot) i++;
        while (fitness[indices[j]] > pivot) j--;
        if (i <= j) {
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
            i++;
            j--;
        }
    }
    
    quicksort_indices(fitness, indices, left, j);
    quicksort_indices(fitness, indices, i, right);
}

// Initialize habitats randomly within bounds
void bbo_initialize_habitats(Optimizer *opt, BBOData *data) {
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Entering bbo_initialize_habitats\n");
#endif
    if (!opt || !opt->population || !opt->bounds || !data || !data->population_data) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Invalid opt, population, bounds, or population_data\n");
#endif
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = data->population_data + i * opt->dim;
        opt->population[i].position = pos;
        if (opt->population[i].position != pos) {
#ifndef NDEBUG
            fprintf(stderr, "WARNING: opt->population[%d].position was preallocated, overwriting with population_data\n", i);
#endif
        }
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = lcg_rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Exiting bbo_initialize_habitats\n");
#endif
}

// Combined migration, mutation, and evaluation phase
void bbo_migration_phase(Optimizer *opt, BBOData *data, double (*objective_function)(double *)) {
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Entering bbo_migration_phase\n");
#endif
    if (!opt || !data || !opt->population || !data->mu || !data->lambda_ || !data->ep_buffer || !data->random_buffer || !objective_function) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Invalid opt, data, buffers, or objective_function\n");
#endif
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        
        // Prepare emigration probabilities
        double ep_sum = 0.0;
        for (int k = 0; k < opt->population_size; k++) {
            data->ep_buffer[k] = (k == i) ? 0.0 : data->mu[k];
            ep_sum += data->ep_buffer[k];
        }
        if (ep_sum > 0.0) {
            for (int k = 0; k < opt->population_size; k++) {
                data->ep_buffer[k] /= ep_sum;
            }
        }
        
        // Migration, mutation, and evaluation
        for (int j = 0; j < opt->dim; j++) {
            // Migration
            if (ep_sum > 0.0 && lcg_rand_double(0.0, 1.0) <= data->lambda_[i]) {
                int source_idx = roulette_wheel_selection(data->ep_buffer, opt->population_size);
                if (source_idx >= 0 && source_idx < opt->population_size) {
                    double *source_pos = opt->population[source_idx].position;
                    pos[j] += BBO_ALPHA * (source_pos[j] - pos[j]);
                }
            }
            
            // Mutation
            if (lcg_rand_double(0.0, 1.0) <= BBO_MUTATION_PROB) {
                if (data->random_buffer_idx >= data->random_buffer_size) {
                    data->random_buffer_idx = 0; // Wrap around
                }
                pos[j] += data->mutation_sigma * data->random_buffer[data->random_buffer_idx++];
            }
        }
        
        // Evaluate fitness
        opt->population[i].fitness = objective_function(pos);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
        }
    }
    enforce_bound_constraints(opt);
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Exiting bbo_migration_phase\n");
#endif
}

// Selection phase with custom quicksort
void bbo_selection_phase(Optimizer *opt, BBOData *data) {
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Entering bbo_selection_phase\n");
#endif
    if (!opt || !opt->population || !data || !data->population_data) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Invalid opt, population, or population_data\n");
#endif
        exit(1);
    }
    
    // Create index and fitness arrays
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    if (!indices || !fitness) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Failed to allocate indices or fitness\n");
#endif
        free(indices);
        free(fitness);
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
        fitness[i] = opt->population[i].fitness;
    }
    
    // Sort indices by fitness
    quicksort_indices(fitness, indices, 0, opt->population_size - 1);
    
    // Keep best n_keep and replace others
    int n_keep = (int)(KEEP_RATE * opt->population_size);
    
    // Allocate temporary array for positions
    double *temp_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    if (!temp_positions) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Failed to allocate temp_positions\n");
#endif
        free(indices);
        free(fitness);
        exit(1);
    }
    
    // Copy sorted population
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(&temp_positions[i * opt->dim], opt->population[indices[i]].position, opt->dim * sizeof(double));
        fitness[i] = opt->population[indices[i]].fitness;
    }
    
    // Copy back to population_data and update pointers
    memcpy(data->population_data, temp_positions, opt->population_size * opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].position = data->population_data + i * opt->dim;
        opt->population[i].fitness = fitness[i];
    }
    
    // Clean up
    free(temp_positions);
    free(fitness);
    free(indices);
    enforce_bound_constraints(opt);
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Exiting bbo_selection_phase\n");
#endif
}

// Main Optimization Function
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Entering BBO_optimize\n");
#endif
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position || !opt->bounds) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Invalid optimizer or objective function\n");
#endif
        exit(1);
    }
    
    // Initialize BBO-specific data
    BBOData *data = (BBOData *)malloc(sizeof(BBOData));
    if (!data) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Failed to allocate BBOData\n");
#endif
        exit(1);
    }
    
    // Allocate contiguous memory
    data->mu = (double *)malloc(opt->population_size * sizeof(double));
    data->lambda_ = (double *)malloc(opt->population_size * sizeof(double));
    data->ep_buffer = (double *)malloc(opt->population_size * sizeof(double));
    data->population_data = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    data->random_buffer_size = opt->population_size * opt->dim; // Enough for one iteration
    data->random_buffer = (double *)malloc(data->random_buffer_size * sizeof(double));
    data->mutation_sigma = MUTATION_SCALE * (opt->bounds[1] - opt->bounds[0]);
    data->random_buffer_idx = 0;
    
    if (!data->mu || !data->lambda_ || !data->ep_buffer || !data->population_data || !data->random_buffer) {
#ifndef NDEBUG
        fprintf(stderr, "ERROR: Failed to allocate mu, lambda_, ep_buffer, population_data, or random_buffer\n");
#endif
        free(data->mu);
        free(data->lambda_);
        free(data->ep_buffer);
        free(data->population_data);
        free(data->random_buffer);
        free(data);
        exit(1);
    }
    
    // Precompute Gaussian random numbers
    for (int i = 0; i < data->random_buffer_size; i += 2) {
        double n1, n2;
        generate_normal_pair(&n1, &n2);
        data->random_buffer[i] = n1;
        if (i + 1 < data->random_buffer_size) {
            data->random_buffer[i + 1] = n2;
        }
    }
    
    compute_migration_rates(data->mu, data->lambda_, opt->population_size);
    
    bbo_initialize_habitats(opt, data);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
#ifndef NDEBUG
        fprintf(stderr, "DEBUG: Iteration %d\n", iter + 1);
#endif
        bbo_migration_phase(opt, data, objective_function);
        bbo_selection_phase(opt, data);
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    // Clean up
    free(data->random_buffer);
    free(data->population_data);
    free(data->ep_buffer);
    free(data->mu);
    free(data->lambda_);
    free(data);
#ifndef NDEBUG
    fprintf(stderr, "DEBUG: Exiting BBO_optimize\n");
#endif
}
