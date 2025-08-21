#include "KCA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Precompute bit scales for binary-to-continuous conversion
double BIT_SCALES[32] = {0};

// Initialize BIT_SCALES at program start
__attribute__((constructor)) void init_bit_scales() {
    for (int b = 0; b < 32; b++) {
        BIT_SCALES[b] = 1.0 / (1 << (b + 1));
    }
}

// 🌟 Inline random double generator
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🌟 Optimized binary to continuous conversion
void kca_binary_to_continuous(const int* binary_key, double* continuous_key, int key_length, int dim) {
    int bits_per_dim = key_length / dim;
    for (int d = 0; d < dim; d++) {
        double value = 0.0;
        const int* key_ptr = binary_key + d * bits_per_dim;
        for (int b = 0; b < bits_per_dim; b++) {
            value += key_ptr[b] * BIT_SCALES[b];
        }
        continuous_key[d] = SHUBERT_MIN + value * (SHUBERT_MAX - SHUBERT_MIN);
    }
}

// 🌱 Initialize keys with random binary values
void kca_initialize_keys(Optimizer* opt, int* keys, int key_length) {
    int total_bits = opt->population_size * key_length;
    for (int j = 0; j < total_bits; j++) {
        keys[j] = (rand_double(0.0, 1.0) < KEY_INIT_PROB) ? 1 : 0;
    }
    #ifdef DEBUG
    for (int i = 0; i < opt->population_size; i++) {
        printf("Initial key %d: ", i);
        for (int j = 0; j < key_length; j++) {
            printf("%d ", keys[i * key_length + j]);
        }
        printf("\n");
    }
    #endif
}

// 📊 Evaluate fitness of all keys
void kca_evaluate_keys(Optimizer* opt, int* keys, int key_length, double (*objective_function)(double*)) {
    double* temp_position = (double*)malloc(opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        kca_binary_to_continuous(keys + i * key_length, temp_position, key_length, opt->dim);
        opt->population[i].fitness = objective_function(temp_position) * FITNESS_SCALE;
        #ifdef DEBUG
        printf("Key %d: Fitness: %f\n", i, opt->population[i].fitness);
        #endif
    }
    free(temp_position);
}

// 🔍 Optimized probability factor calculation
void kca_calculate_probability_factor(Optimizer* opt, const int* keys, int key_length, double* prob_factors) {
    int half_pop = opt->population_size / 2;
    for (int j = 0; j < key_length; j++) {
        int tooth_sum = 0;
        for (int i = 0; i < half_pop; i++) {
            tooth_sum += keys[i * key_length + j];
        }
        double average = (double)tooth_sum / half_pop;
        double prob = 1.0 - average;
        for (int i = 0; i < half_pop; i++) {
            prob_factors[i * key_length + j] = prob;
        }
    }
}

// 🔧 Optimized new key generation
void kca_generate_new_keys(Optimizer* opt, int* keys, int key_length, const double* prob_factors, int use_kca1) {
    int half_pop = opt->population_size / 2;
    for (int i = 0; i < half_pop; i++) {
        int* new_key = keys + (half_pop + i) * key_length;
        const int* old_key = keys + i * key_length;
        const double* prob = prob_factors + i * key_length;
        for (int j = 0; j < key_length; j++) {
            new_key[j] = old_key[j];
            double random_num = rand_double(0.0, 1.0);
            if ((use_kca1 && random_num < prob[j]) || (!use_kca1 && random_num > prob[j])) {
                new_key[j] = 1 - new_key[j];
            }
        }
    }
}

// 🌟 Static fitness array for qsort comparison
static double* fitness_for_compare = NULL;

// 🌟 Comparison function for qsort
static int compare_fitness(const void* a, const void* b) {
    double fa = fitness_for_compare[*(const int*)a];
    double fb = fitness_for_compare[*(const int*)b];
    return (fa > fb) - (fa < fb);
}

// 🚀 Main KCA Optimization Function
void KCA_optimize(void* optimizer, ObjectiveFunction objective_function) {
    Optimizer* opt = (Optimizer*)optimizer;
    srand(42);

    int key_length = opt->dim * BITS_PER_DIM;
    int use_kca1 = 1;
    int half_pop = opt->population_size / 2;

    // Allocate memory
    int* keys = (int*)malloc(opt->population_size * key_length * sizeof(int));
    double* prob_factors = (double*)malloc(half_pop * key_length * sizeof(double));
    double* fitness = (double*)malloc(opt->population_size * sizeof(double));
    int* sorted_indices = (int*)malloc(opt->population_size * sizeof(int));
    int* temp_keys = (int*)malloc(opt->population_size * key_length * sizeof(int));

    // Initialize keys
    kca_initialize_keys(opt, keys, key_length);

    // Main optimization loop
    for (int generation = 0; generation < opt->max_iter; generation++) {
        // Evaluate fitness
        kca_evaluate_keys(opt, keys, key_length, objective_function);

        // Update best solution
        int min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
        }
        if (opt->population[min_idx].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[min_idx].fitness;
            kca_binary_to_continuous(keys + min_idx * key_length, opt->best_solution.position, key_length, opt->dim);
        }

        // Sort keys by fitness
        for (int i = 0; i < opt->population_size; i++) {
            sorted_indices[i] = i;
            fitness[i] = opt->population[i].fitness;
        }
        fitness_for_compare = fitness; // Set fitness array for comparison
        qsort(sorted_indices, opt->population_size, sizeof(int), compare_fitness);
        fitness_for_compare = NULL; // Reset to avoid accidental use

        // Reorganize keys
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(temp_keys + i * key_length,
                   keys + sorted_indices[i] * key_length,
                   key_length * sizeof(int));
            kca_binary_to_continuous(temp_keys + i * key_length,
                                  opt->population[i].position,
                                  key_length,
                                  opt->dim);
            opt->population[i].fitness = fitness[sorted_indices[i]];
        }
        memcpy(keys, temp_keys, opt->population_size * key_length * sizeof(int));

        // Calculate probability factors
        kca_calculate_probability_factor(opt, keys, key_length, prob_factors);

        // Generate new keys
        kca_generate_new_keys(opt, keys, key_length, prob_factors, use_kca1);

        // Update population positions
        for (int i = 0; i < opt->population_size; i++) {
            kca_binary_to_continuous(keys + i * key_length,
                                  opt->population[i].position,
                                  key_length,
                                  opt->dim);
        }

        // Enforce bounds
        enforce_bound_constraints(opt);

        // Print progress sparingly
        if (generation % 10 == 0) {
            printf("Iteration %d: Best Fitness = %f\n", generation + 1, opt->best_solution.fitness);
        }
    }

    // Free allocated memory
    free(keys);
    free(prob_factors);
    free(fitness);
    free(sorted_indices);
    free(temp_keys);
}
