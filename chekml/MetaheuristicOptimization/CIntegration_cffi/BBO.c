#include "BBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
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
    double r = rand_double(0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += probabilities[i];
        if (r <= cumsum) return i;
    }
    return size - 1;
}

// Generate Gaussian random number using Box-Muller transform
static inline double generate_normal() {
    static double cache = 0.0;
    static int has_cache = 0;
    
    if (has_cache) {
        has_cache = 0;
        return cache;
    }
    
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    cache = r * sin(theta);
    has_cache = 1;
    return r * cos(theta);
}

// Initialize habitats randomly within bounds
void bbo_initialize_habitats(Optimizer *opt, BBOData *data) {
    fprintf(stderr, "DEBUG: Entering bbo_initialize_habitats\n");
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "ERROR: Invalid opt, population, or bounds\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "ERROR: Population[%d].position is NULL\n", i);
            exit(1);
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "DEBUG: Exiting bbo_initialize_habitats\n");
}

// Combined migration and mutation phase
void bbo_migration_phase(Optimizer *opt, BBOData *data) {
    fprintf(stderr, "DEBUG: Entering bbo_migration_phase\n");
    if (!opt || !data || !opt->population || !data->mu || !data->lambda_ || !data->ep_buffer) {
        fprintf(stderr, "ERROR: Invalid opt, data, or buffers\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
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
        
        // Migration and mutation
        for (int j = 0; j < opt->dim; j++) {
            // Migration
            if (ep_sum > 0.0 && rand_double(0.0, 1.0) <= data->lambda_[i]) {
                int source_idx = roulette_wheel_selection(data->ep_buffer, opt->population_size);
                if (source_idx >= 0 && source_idx < opt->population_size && opt->population[source_idx].position) {
                    opt->population[i].position[j] += BBO_ALPHA * (opt->population[source_idx].position[j] - opt->population[i].position[j]);
                }
            }
            
            // Mutation
            if (rand_double(0.0, 1.0) <= MUTATION_PROB) {
                opt->population[i].position[j] += data->mutation_sigma * generate_normal();
            }
        }
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "DEBUG: Exiting bbo_migration_phase\n");
}

// Context for qsort comparison
static struct {
    Optimizer *opt;
    int *indices;
} global_sort_ctx;

// Comparison function for qsort
static int compare_habitats(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    double fitness_a = global_sort_ctx.opt->population[idx_a].fitness;
    double fitness_b = global_sort_ctx.opt->population[idx_b].fitness;
    return (fitness_a > fitness_b) - (fitness_a < fitness_b);
}

// Selection phase using qsort
void bbo_selection_phase(Optimizer *opt, BBOData *data) {
    fprintf(stderr, "DEBUG: Entering bbo_selection_phase\n");
    if (!opt || !opt->population) {
        fprintf(stderr, "ERROR: Invalid opt or population\n");
        exit(1);
    }
    
    // Create index array
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "ERROR: Failed to allocate indices\n");
        exit(1);
    }
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Sort indices by fitness
    global_sort_ctx.opt = opt;
    global_sort_ctx.indices = indices;
    qsort(indices, opt->population_size, sizeof(int), compare_habitats);
    
    // Keep best n_keep and replace others
    int n_keep = (int)(KEEP_RATE * opt->population_size);
    int n_new = opt->population_size - n_keep;
    
    // Allocate temporary array for positions
    double *temp_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    if (!temp_positions || !temp_fitness) {
        fprintf(stderr, "ERROR: Failed to allocate temp arrays\n");
        free(indices);
        if (temp_positions) free(temp_positions);
        if (temp_fitness) free(temp_fitness);
        exit(1);
    }
    
    // Copy sorted population
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(&temp_positions[i * opt->dim], opt->population[indices[i]].position, opt->dim * sizeof(double));
        temp_fitness[i] = opt->population[indices[i]].fitness;
    }
    
    // Copy back to population
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(opt->population[i].position, &temp_positions[i * opt->dim], opt->dim * sizeof(double));
        opt->population[i].fitness = temp_fitness[i];
    }
    
    // Clean up
    free(temp_positions);
    free(temp_fitness);
    free(indices);
    enforce_bound_constraints(opt);
    fprintf(stderr, "DEBUG: Exiting bbo_selection_phase\n");
}

// Main Optimization Function
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    fprintf(stderr, "DEBUG: Entering BBO_optimize\n");
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position || !opt->bounds) {
        fprintf(stderr, "ERROR: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    // Initialize BBO-specific data
    BBOData *data = (BBOData *)malloc(sizeof(BBOData));
    if (!data) {
        fprintf(stderr, "ERROR: Failed to allocate BBOData\n");
        exit(1);
    }
    data->mu = (double *)malloc(opt->population_size * sizeof(double));
    data->lambda_ = (double *)malloc(opt->population_size * sizeof(double));
    data->ep_buffer = (double *)malloc(opt->population_size * sizeof(double));
    data->mutation_sigma = MUTATION_SCALE * (opt->bounds[1] - opt->bounds[0]);
    data->store_history = 0; // Disable history for debugging
    data->history = NULL;
    
    if (!data->mu || !data->lambda_ || !data->ep_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate mu, lambda_, or ep_buffer\n");
        free(data->mu);
        free(data->lambda_);
        free(data->ep_buffer);
        free(data);
        exit(1);
    }
    
    if (data->store_history) {
        data->history = (void *)malloc(opt->max_iter * sizeof(*data->history));
        if (!data->history) {
            fprintf(stderr, "ERROR: Failed to allocate history\n");
            free(data->mu);
            free(data->lambda_);
            free(data->ep_buffer);
            free(data);
            exit(1);
        }
        for (int i = 0; i < opt->max_iter; i++) {
            data->history[i].solution = (double *)malloc(opt->dim * sizeof(double));
            if (!data->history[i].solution) {
                fprintf(stderr, "ERROR: Failed to allocate history[%d].solution\n", i);
                for (int j = 0; j < i; j++) free(data->history[j].solution);
                free(data->history);
                free(data->mu);
                free(data->lambda_);
                free(data->ep_buffer);
                free(data);
                exit(1);
            }
        }
    }
    
    compute_migration_rates(data->mu, data->lambda_, opt->population_size);
    
    bbo_initialize_habitats(opt, data);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        fprintf(stderr, "DEBUG: Iteration %d\n", iter + 1);
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            if (!opt->population[i].position) {
                fprintf(stderr, "ERROR: Population[%d].position is NULL\n", i);
                exit(1);
            }
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }
        
        bbo_migration_phase(opt, data);
        
        // Re-evaluate after migration and mutation
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }
        
        bbo_selection_phase(opt, data);
        
        // Store history if enabled
        if (data->store_history) {
            data->history[iter].iteration = iter;
            data->history[iter].value = opt->best_solution.fitness;
            memcpy(data->history[iter].solution, opt->best_solution.position, opt->dim * sizeof(double));
        }
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    // Clean up
    if (data->store_history && data->history) {
        for (int i = 0; i < opt->max_iter; i++) {
            free(data->history[i].solution);
        }
        free(data->history);
    }
    free(data->ep_buffer);
    free(data->mu);
    free(data->lambda_);
    free(data);
    fprintf(stderr, "DEBUG: Exiting BBO_optimize\n");
}
