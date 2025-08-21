#include "FSA.h"
#include <stdlib.h>  // For posix_memalign, free
#include <string.h>  // For memcpy
#include <math.h>    // For fmin, fmax, INFINITY
#include <time.h>    // For seeding RNG
#include <limits.h>  // For UINT_MAX

// Additional data for FSA
typedef struct {
    double *local_best_positions;  // Contiguous: population_size * dim
    double *local_best_values;     // Size: population_size
    double *rand_buffer;           // Precomputed random values
    unsigned int rng_state;        // Xorshift state
} FSAData;

// Fast Xorshift RNG
static inline double xorshift_double(FSAData *data, double min, double max) {
    data->rng_state ^= data->rng_state << 13;
    data->rng_state ^= data->rng_state >> 17;
    data->rng_state ^= data->rng_state << 5;
    return min + (max - min) * ((double)data->rng_state / UINT_MAX);
}

// Precompute random values
static inline void precompute_random(FSAData *data, int count, double min, double max) {
    double *rand_buf = data->rand_buffer;
    for (int i = 0; i < count; i++) {
        rand_buf[i] = xorshift_double(data, min, max);
    }
}

// Internal helper functions
static void fsa_initialize_population(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function, double *temp_solution) {
    int i, j;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;
    double *positions = opt->population[0].position;
    double *local_best = data->local_best_positions;

    // Initialize RNG state
    data->rng_state = (unsigned int)time(NULL);

    // Precompute random values for initialization
    precompute_random(data, pop_size * dim, 0.0, 1.0);
    int rand_idx = 0;

    // Initialize population and local bests
    #pragma omp simd
    for (i = 0; i < pop_size * dim; i++) {
        int pop_idx = i / dim;
        int dim_idx = i % dim;
        double range = upper_bounds[2 * dim_idx] - lower_bounds[2 * dim_idx];
        positions[i] = lower_bounds[2 * dim_idx] + data->rand_buffer[rand_idx++] * range;
        local_best[i] = positions[i];
    }

    // Evaluate fitness
    for (i = 0; i < pop_size; i++) {
        double fitness = objective_function(&positions[i * dim]);
        opt->population[i].fitness = fitness;
        data->local_best_values[i] = fitness;
    }

    // Find initial global best
    int best_idx = 0;
    for (i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < opt->population[best_idx].fitness) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[best_idx].fitness;
    memcpy(opt->best_solution.position, &positions[best_idx * dim], dim * sizeof(double));

    enforce_bound_constraints(opt);
}

// Internal helper function for population update
static void fsa_update_population(Optimizer *opt, double **d_move, double **x_move, FSAData *data, ObjectiveFunction objective_function, double *temp_solution) {
    int i, j;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;
    double *global_best = opt->best_solution.position;
    double *positions = opt->population[0].position;
    double *local_best = data->local_best_positions;

    // Precompute random values
    precompute_random(data, pop_size * dim * 2, 0.0, 1.0);
    int rand_idx = 0;

    for (i = 0; i < pop_size; i++) {
        double *current_pos = &positions[i * dim];
        double *current_local = &local_best[i * dim];
        double new_fitness;

        // Update position
        #pragma omp simd
        for (j = 0; j < dim; j++) {
            double delta_global = (global_best[j] - current_pos[j]) * data->rand_buffer[rand_idx++];
            double delta_local = (current_local[j] - current_pos[j]) * data->rand_buffer[rand_idx++];
            temp_solution[j] = current_pos[j] + delta_global + delta_local;
            temp_solution[j] = fmax(lower_bounds[2 * j], fmin(upper_bounds[2 * j], temp_solution[j]));
        }

        // Evaluate new solution
        new_fitness = objective_function(temp_solution);

        // Update local best
        if (new_fitness <= data->local_best_values[i]) {
            memcpy(current_local, temp_solution, dim * sizeof(double));
            data->local_best_values[i] = new_fitness;
        }

        // Update global best
        if (new_fitness <= opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(global_best, temp_solution, dim * sizeof(double));
        }

        // Update population
        memcpy(current_pos, temp_solution, dim * sizeof(double));
        opt->population[i].fitness = new_fitness;
    }

    enforce_bound_constraints(opt);
}

// Public interface for population update
void update_population_fsa(Optimizer *opt, double **d_move, double **x_move) {
    exit(1);  // Should be called via FSA_optimize
}

// Internal helper function for initial strategy update
static void fsa_update_with_initial_strategy(Optimizer *opt, FSAData *data, ObjectiveFunction objective_function, double *temp_solution, double *temp_fitness) {
    int i, j;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *lower_bounds = opt->bounds;
    double *upper_bounds = opt->bounds + 1;
    double *global_best = opt->best_solution.position;
    double *positions = opt->population[0].position;
    double *local_best = data->local_best_positions;

    // Precompute random values
    precompute_random(data, pop_size * dim, 0.0, 1.0);
    int rand_idx = 0;

    for (i = 0; i < pop_size; i++) {
        double *current_pos = &positions[i * dim];
        double *current_local = &local_best[i * dim];

        // Update position
        #pragma omp simd
        for (j = 0; j < dim; j++) {
            temp_solution[j] = global_best[j] + (global_best[j] - current_local[j]) * data->rand_buffer[rand_idx++];
            temp_solution[j] = fmax(lower_bounds[2 * j], fmin(upper_bounds[2 * j], temp_solution[j]));
        }

        // Evaluate new solution
        temp_fitness[i] = objective_function(temp_solution);

        // Update if better
        if (temp_fitness[i] <= data->local_best_values[i]) {
            memcpy(current_pos, temp_solution, dim * sizeof(double));
            memcpy(current_local, temp_solution, dim * sizeof(double));
            data->local_best_values[i] = temp_fitness[i];
            opt->population[i].fitness = temp_fitness[i];
        }
    }

    // Update global best
    int best_idx = 0;
    for (i = 1; i < pop_size; i++) {
        if (temp_fitness[i] < temp_fitness[best_idx]) {
            best_idx = i;
        }
    }
    if (temp_fitness[best_idx] <= opt->best_solution.fitness) {
        opt->best_solution.fitness = temp_fitness[best_idx];
        memcpy(opt->best_solution.position, &positions[best_idx * dim], dim * sizeof(double));
    }

    enforce_bound_constraints(opt);
}

// Public interface for initial strategy update
void update_with_initial_strategy(Optimizer *opt) {
    exit(1);  // Should be called via FSA_optimize
}

// Main Optimization Function
void FSA_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int run, iter;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_scores;
    double *best_positions;  // Contiguous: FSA_NUM_RUNS * dim
    double *temp_solution;
    double *temp_fitness;
    double *rand_buffer;

    // Allocate memory with error checking
    if (posix_memalign((void**)&best_scores, 64, FSA_NUM_RUNS * sizeof(double)) != 0 ||
        posix_memalign((void**)&best_positions, 64, FSA_NUM_RUNS * dim * sizeof(double)) != 0 ||
        posix_memalign((void**)&temp_solution, 64, dim * sizeof(double)) != 0 ||
        posix_memalign((void**)&temp_fitness, 64, pop_size * sizeof(double)) != 0 ||
        posix_memalign((void**)&rand_buffer, 64, pop_size * dim * 2 * sizeof(double)) != 0) {
        exit(1);  // Allocation failure
    }

    for (run = 0; run < FSA_NUM_RUNS; run++) {
        // Allocate FSAData
        FSAData data;
        if (posix_memalign((void**)&data.local_best_positions, 64, pop_size * dim * sizeof(double)) != 0 ||
            posix_memalign((void**)&data.local_best_values, 64, pop_size * sizeof(double)) != 0) {
            free(best_scores); free(best_positions); free(temp_solution); free(temp_fitness); free(rand_buffer);
            exit(1);  // Allocation failure
        }
        data.rand_buffer = rand_buffer;

        fsa_initialize_population(opt, &data, objective_function, temp_solution);

        for (iter = 0; iter < opt->max_iter; iter++) {
            fsa_update_population(opt, NULL, NULL, &data, objective_function, temp_solution);
            fsa_update_with_initial_strategy(opt, &data, objective_function, temp_solution, temp_fitness);
        }

        best_scores[run] = opt->best_solution.fitness;
        memcpy(&best_positions[run * dim], opt->best_solution.position, dim * sizeof(double));

        // Clean up FSAData
        free(data.local_best_positions);
        free(data.local_best_values);
    }

    // Find best result
    int best_run = 0;
    for (run = 1; run < FSA_NUM_RUNS; run++) {
        if (best_scores[run] < best_scores[best_run]) {
            best_run = run;
        }
    }
    opt->best_solution.fitness = best_scores[best_run];
    memcpy(opt->best_solution.position, &best_positions[best_run * dim], dim * sizeof(double));

    // Clean up
    free(best_scores);
    free(best_positions);
    free(temp_solution);
    free(temp_fitness);
    free(rand_buffer);
}
