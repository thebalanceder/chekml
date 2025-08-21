#include "MFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>  // Added for time()

// Inline fast random number generator (Xorshift)
static inline unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline double fast_rand_double(unsigned int *state, double min, double max) {
    return min + (max - min) * (xorshift32(state) / 4294967295.0);
}

// Global context for qsort comparison (to avoid qsort_r)
static double *g_fitness = NULL;

// Create MFOData structure
MFOData* mfo_create_data(int population_size, int dim) {
    MFOData *mfo_data = (MFOData *)malloc(sizeof(MFOData));
    mfo_data->objective_function = NULL;
    mfo_data->iteration = 0;
    mfo_data->rng_state = (unsigned int)time(NULL);  // Now safe with time.h included

    // Allocate arrays
    mfo_data->population = (Solution *)malloc(population_size * sizeof(Solution));
    mfo_data->fitness = (double *)malloc(population_size * sizeof(double));
    mfo_data->best_flames = (Solution *)malloc(population_size * sizeof(Solution));
    mfo_data->best_flame_fitness = (double *)malloc(population_size * sizeof(double));
    mfo_data->indices = (int *)malloc(population_size * sizeof(int));
    mfo_data->spiral_cache = (double *)malloc(population_size * sizeof(double));

    // Allocate position arrays
    for (int i = 0; i < population_size; i++) {
        mfo_data->population[i].position = (double *)malloc(dim * sizeof(double));
        mfo_data->best_flames[i].position = (double *)malloc(dim * sizeof(double));
        mfo_data->fitness[i] = INFINITY;
        mfo_data->best_flame_fitness[i] = INFINITY;
    }

    return mfo_data;
}

// Free MFOData structure
void mfo_free_data(MFOData *mfo_data, int population_size) {
    for (int i = 0; i < population_size; i++) {
        free(mfo_data->population[i].position);
        free(mfo_data->best_flames[i].position);
    }
    free(mfo_data->population);
    free(mfo_data->fitness);
    free(mfo_data->best_flames);
    free(mfo_data->best_flame_fitness);
    free(mfo_data->indices);
    free(mfo_data->spiral_cache);
    free(mfo_data);
}

// Comparison function for qsort
static int compare_fitness_mfo(const void *a, const void *b) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    return (g_fitness[idx_a] > g_fitness[idx_b]) - (g_fitness[idx_a] < g_fitness[idx_b]);
}

// Sort population by fitness
void mfo_sort_population(MFOData *restrict mfo_data, int population_size, int dim) {
    g_fitness = mfo_data->fitness;  // Set global context for qsort
    for (int i = 0; i < population_size; i++) {
        mfo_data->indices[i] = i;
    }
    qsort(mfo_data->indices, population_size, sizeof(int), compare_fitness_mfo);
    g_fitness = NULL;  // Reset global context
}

// Update flames
void mfo_update_flames(MFOData *restrict mfo_data, int population_size, int dim, int max_iter) {
    // Compute fitness
    for (int i = 0; i < population_size; i++) {
        mfo_data->fitness[i] = mfo_data->objective_function(mfo_data->population[i].position);
    }

    // Sort population
    mfo_sort_population(mfo_data, population_size, dim);

    // Update best flames
    if (mfo_data->iteration == 0 || mfo_data->fitness[mfo_data->indices[0]] < mfo_data->best_flame_fitness[0]) {
        for (int i = 0; i < population_size; i++) {
            int idx = mfo_data->indices[i];
            memcpy(mfo_data->best_flames[i].position, mfo_data->population[idx].position, dim * sizeof(double));
            mfo_data->best_flame_fitness[i] = mfo_data->fitness[idx];
        }
    }
}

// Update moth positions
void mfo_update_moth_positions(MFOData *restrict mfo_data, int population_size, int dim, int max_iter) {
    // Calculate number of flames
    int flame_no = (int)(population_size - mfo_data->iteration * ((population_size - 1.0) / max_iter));

    // Non-linear decay for 'a'
    double t = (double)mfo_data->iteration / max_iter;
    double a = MFO_A_INITIAL - (MFO_A_INITIAL - MFO_A_FINAL) * (1.0 - exp(-3.0 * t));

    // Precompute spiral values
    for (int i = 0; i < population_size; i++) {
        double t_rand = (a - 1.0) * fast_rand_double(&mfo_data->rng_state, 0.0, 1.0) + 1.0;
        mfo_data->spiral_cache[i] = exp(MFO_B_CONSTANT * t_rand) * cos(t_rand * TWO_PI);
    }

    // Update positions
    for (int i = 0; i < population_size; i++) {
        int flame_idx = (i < flame_no) ? i : (flame_no - 1);
        double *moth_pos = mfo_data->population[i].position;
        double *flame_pos = mfo_data->best_flames[flame_idx].position;
        double spiral = mfo_data->spiral_cache[i];

        for (int j = 0; j < dim; j++) {
            double distance = fabs(flame_pos[j] - moth_pos[j]);
            moth_pos[j] = distance * spiral + flame_pos[j];
        }
    }
}

// Main Optimization Function
void MFO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    MFOData *mfo_data = mfo_create_data(opt->population_size, opt->dim);
    mfo_data->objective_function = objective_function;
    mfo_data->iteration = 0;

    // Initialize population fitness
    double best_fitness = INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        mfo_data->fitness[i] = objective_function(opt->population[i].position);
        if (mfo_data->fitness[i] < best_fitness) {
            best_fitness = mfo_data->fitness[i];
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));

    printf("MFO is optimizing your problem\n");
    printf("Initial best solution: pos=[%f, %f], fitness=%f\n",
           opt->best_solution.position[0], opt->best_solution.position[1], best_fitness);

    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        mfo_data->iteration = iter;

        // Update flames and positions
        mfo_update_flames(mfo_data, opt->population_size, opt->dim, opt->max_iter);
        mfo_update_moth_positions(mfo_data, opt->population_size, opt->dim, opt->max_iter);

        // Enforce bounds
        enforce_bound_constraints(opt);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            mfo_data->fitness[i] = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                printf("Updated best solution at iteration %d: fitness=%f, pos=[%f, %f]\n",
                       iter, new_fitness, opt->best_solution.position[0], opt->best_solution.position[1]);
            }
        }

        // Log progress
        if (iter % 100 == 0) {
            printf("At iteration %d, the best fitness is %f\n", iter, opt->best_solution.fitness);
        }
    }

    // Clean up
    mfo_free_data(mfo_data, opt->population_size);
}
