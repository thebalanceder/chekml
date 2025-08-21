#include "ALO.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>
#include <emmintrin.h>  // For SSE2 intrinsics (cache alignment)

// Optimized quicksort with median-of-three pivot
static void quicksort_with_indices(double *arr, int *indices, int low, int high) {
    if (low < high) {
        // Median-of-three pivot selection
        int mid = low + (high - low) / 2;
        if (arr[low] > arr[mid]) {
            double temp = arr[low]; arr[low] = arr[mid]; arr[mid] = temp;
            int temp_idx = indices[low]; indices[low] = indices[mid]; indices[mid] = temp_idx;
        }
        if (arr[low] > arr[high]) {
            double temp = arr[low]; arr[low] = arr[high]; arr[high] = temp;
            int temp_idx = indices[low]; indices[low] = indices[high]; indices[high] = temp_idx;
        }
        if (arr[mid] > arr[high]) {
            double temp = arr[mid]; arr[mid] = arr[high]; arr[high] = temp;
            int temp_idx = indices[mid]; indices[mid] = indices[high]; indices[high] = temp_idx;
        }

        double pivot = arr[mid];
        double temp = arr[mid]; arr[mid] = arr[high]; arr[mid] = temp;
        int temp_idx = indices[mid]; indices[mid] = indices[high]; indices[high] = temp_idx;

        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
                temp_idx = indices[i]; indices[i] = indices[j]; indices[j] = temp_idx;
            }
        }

        temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;
        temp_idx = indices[i + 1]; indices[i + 1] = indices[high]; indices[high] = temp_idx;

        int pi = i + 1;
        quicksort_with_indices(arr, indices, low, pi - 1);
        quicksort_with_indices(arr, indices, pi + 1, high);
    }
}

// Initialize populations
void initialize_populations(Optimizer *opt, double *antlion_positions, uint32_t *rng_state) {
    #pragma omp simd
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = alo_rand_double(rng_state, lb, ub);
            antlion_positions[i * opt->dim + j] = pos[j];
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Roulette wheel selection
int roulette_wheel_selection(double *weights, int size, uint32_t *rng_state) {
    double accumulation = 0.0;
    #pragma omp simd reduction(+:accumulation)
    for (int i = 0; i < size; i++) {
        accumulation += weights[i];
    }

    double p = alo_rand_double(rng_state, 0.0, accumulation);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += weights[i];
        if (cumsum > p) return i;
    }
    return 0;
}

// Random walk phase (full walk for accuracy, optimized for cache)
void random_walk_phase(Optimizer *opt, int t, double *antlion, double *walk_buffer, uint32_t *rng_state) {
    double I = 1.0;
    double T = (double)opt->max_iter;

    // Adjust I based on iteration
    if (t > T / 10) I = 1.0 + I_FACTOR_1 * (t / T);
    if (t > T / 2) I = 1.0 + I_FACTOR_2 * (t / T);
    if (t > T * 3 / 4) I = 1.0 + I_FACTOR_3 * (t / T);
    if (t > T * 0.9) I = 1.0 + I_FACTOR_4 * (t / T);
    if (t > T * 0.95) I = 1.0 + I_FACTOR_5 * (t / T);

    // Compute bounds
    double lb[opt->dim], ub[opt->dim];
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        lb[j] = opt->bounds[2 * j] / I;
        ub[j] = opt->bounds[2 * j + 1] / I;
    }

    // Move interval around antlion
    double r = alo_rand_double(rng_state, 0.0, 1.0);
    if (r < 0.5) {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) lb[j] += antlion[j];
    } else {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) lb[j] = -lb[j] + antlion[j];
    }

    r = alo_rand_double(rng_state, 0.0, 1.0);
    if (r >= 0.5) {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) ub[j] += antlion[j];
    } else {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) ub[j] = -ub[j] + antlion[j];
    }

    // Generate full random walk
    double *X = (double *)_mm_malloc((opt->max_iter + 1) * sizeof(double), CACHE_LINE_SIZE);
    if (!X) return;

    for (int j = 0; j < opt->dim; j++) {
        memset(X, 0, (opt->max_iter + 1) * sizeof(double));
        #pragma omp simd
        for (int k = 0; k < opt->max_iter; k++) {
            r = alo_rand_double(rng_state, 0.0, 1.0);
            X[k + 1] = X[k] + (r > 0.5 ? 1.0 : -1.0);
        }

        double a = X[0], b = X[0];
        #pragma omp simd
        for (int k = 0; k <= opt->max_iter; k++) {
            if (X[k] < a) a = X[k];
            if (X[k] > b) b = X[k];
        }

        double c = lb[j], d = ub[j];
        walk_buffer[j] = ((X[t] - a) * (d - c)) / (b - a + 1e-10) + c;
    }

    _mm_free(X);
}

// Update ant positions
void update_ant_positions(Optimizer *opt, int t, double *antlion_positions, double *walk_buffer, double *weights, uint32_t *rng_state) {
    #pragma omp simd
    for (int i = 0; i < opt->population_size; i++) {
        weights[i] = 1.0 / (opt->population[i].fitness + ROULETTE_EPSILON);
    }

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;

        // Select antlion using roulette wheel
        int roulette_idx = roulette_wheel_selection(weights, opt->population_size, rng_state);
        if (roulette_idx == -1) roulette_idx = 0;

        // Random walk around selected antlion
        random_walk_phase(opt, t, &antlion_positions[roulette_idx * opt->dim], walk_buffer, rng_state);
        double RA[opt->dim];
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) RA[j] = walk_buffer[j];

        // Random walk around elite antlion
        random_walk_phase(opt, t, opt->best_solution.position, walk_buffer, rng_state);
        double RE[opt->dim];
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) RE[j] = walk_buffer[j];

        // Update ant position (Equation 2.13)
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = (RA[j] + RE[j]) / 2.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Update antlions
void update_antlions_phase(Optimizer *opt, double *antlion_positions, double *combined_fitness, int *indices) {
    int total_size = 2 * opt->population_size;

    // Copy fitness values
    #pragma omp simd
    for (int i = 0; i < opt->population_size; i++) {
        combined_fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }
    #pragma omp simd
    for (int i = 0; i < opt->population_size; i++) {
        combined_fitness[opt->population_size + i] = opt->population[i].fitness;
        indices[opt->population_size + i] = opt->population_size + i;
    }

    // Sort using quicksort
    quicksort_with_indices(combined_fitness, indices, 0, total_size - 1);

    // Update antlion positions
    for (int i = 0; i < opt->population_size; i++) {
        int idx = indices[i];
        double *src = (idx < opt->population_size) ? &antlion_positions[idx * opt->dim] : opt->population[idx - opt->population_size].position;
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = src[j];
        }
        opt->population[i].fitness = combined_fitness[i];
    }

    // Update elite
    if (combined_fitness[0] < opt->best_solution.fitness) {
        opt->best_solution.fitness = combined_fitness[0];
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = antlion_positions[j];
        }
    }

    // Ensure elite is in population
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        antlion_positions[j] = opt->best_solution.position[j];
    }
    opt->population[0].fitness = opt->best_solution.fitness;
}

// Main Optimization Function
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed
    uint32_t rng_state = (uint32_t)time(NULL);

    // Pre-allocate buffers
    double *antlion_positions = (double *)_mm_malloc(opt->population_size * opt->dim * sizeof(double), CACHE_LINE_SIZE);
    double *walk_buffer = (double *)_mm_malloc(opt->dim * sizeof(double), CACHE_LINE_SIZE);
    double *combined_fitness = (double *)_mm_malloc(2 * opt->population_size * sizeof(double), CACHE_LINE_SIZE);
    int *indices = (int *)_mm_malloc(2 * opt->population_size * sizeof(int), CACHE_LINE_SIZE);
    double *weights = (double *)_mm_malloc(opt->population_size * sizeof(double), CACHE_LINE_SIZE);
    if (!antlion_positions || !walk_buffer || !combined_fitness || !indices || !weights) {
        _mm_free(antlion_positions); _mm_free(walk_buffer);
        _mm_free(combined_fitness); _mm_free(indices); _mm_free(weights);
        return;
    }

    // Initialize populations
    initialize_populations(opt, antlion_positions, &rng_state);

    // Compute initial fitness
    #pragma omp simd
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Set initial elite
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            #pragma omp simd
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        update_ant_positions(opt, t, antlion_positions, walk_buffer, weights, &rng_state);
        #pragma omp simd
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        update_antlions_phase(opt, antlion_positions, combined_fitness, indices);
    }

    // Clean up
    _mm_free(antlion_positions);
    _mm_free(walk_buffer);
    _mm_free(combined_fitness);
    _mm_free(indices);
    _mm_free(weights);
}
