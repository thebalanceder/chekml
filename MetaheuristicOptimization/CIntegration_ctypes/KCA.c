#include "KCA.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy()

// 🌟 Function to generate a random double between min and max
double rand_double(double min, double max);

// 🌟 Map binary key to continuous values
void kca_binary_to_continuous(int* binary_key, double* continuous_key, int key_length, int dim) {
    int bits_per_dim = key_length / dim;
    for (int d = 0; d < dim; d++) {
        double value = 0.0;
        for (int b = 0; b < bits_per_dim; b++) {
            value += binary_key[d * bits_per_dim + b] * (1.0 / (1 << (b + 1)));
        }
        // Map to [SHUBERT_MIN, SHUBERT_MAX]
        continuous_key[d] = SHUBERT_MIN + value * (SHUBERT_MAX - SHUBERT_MIN);
    }
}

// 🌱 Initialize keys with random binary values
void kca_initialize_keys(Optimizer* opt, int* keys, int key_length) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < key_length; j++) {
            keys[i * key_length + j] = (rand_double(0.0, 1.0) < KEY_INIT_PROB) ? 1 : 0;
        }
        printf("Initial key %d: ", i);
        for (int j = 0; j < key_length; j++) {
            printf("%d ", keys[i * key_length + j]);
        }
        printf("\n");
    }
}

// 📊 Evaluate fitness of all keys
void kca_evaluate_keys(Optimizer* opt, int* keys, int key_length, double (*objective_function)(double*)) {
    double* temp_position = (double*)malloc(opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        kca_binary_to_continuous(&keys[i * key_length], temp_position, key_length, opt->dim);
        opt->population[i].fitness = objective_function(temp_position) * FITNESS_SCALE;
        printf("Key %d: ", i);
        for (int j = 0; j < key_length; j++) {
            printf("%d ", keys[i * key_length + j]);
        }
        printf("Continuous: ");
        for (int j = 0; j < opt->dim; j++) {
            printf("%f ", temp_position[j]);
        }
        printf("Fitness: %f\n", opt->population[i].fitness);
    }
    free(temp_position);
}

// 🔍 Calculate probability factors for key teeth
void kca_calculate_probability_factor(Optimizer* opt, int* keys, int key_length, double** prob_factors) {
    int half_pop = opt->population_size / 2;
    for (int j = 0; j < key_length; j++) {
        double tooth_sum = 0.0;
        for (int i = 0; i < half_pop; i++) {
            tooth_sum += keys[i * key_length + j];
        }
        double average = tooth_sum / half_pop;
        for (int i = 0; i < half_pop; i++) {
            prob_factors[i][j] = 1.0 - average;
            printf("Prob factor [%d][%d]: %f (tooth_sum: %f, avg: %f)\n", i, j, prob_factors[i][j], tooth_sum, average);
        }
    }
}

// 🔧 Generate new keys based on probability factors
void kca_generate_new_keys(Optimizer* opt, int* keys, int key_length, double** prob_factors, int use_kca1) {
    int half_pop = opt->population_size / 2;
    for (int i = 0; i < half_pop; i++) {
        printf("Generating new key %d: ", half_pop + i);
        for (int j = 0; j < key_length; j++) {
            double random_num = rand_double(0.0, 1.0);
            int new_idx = (half_pop + i) * key_length + j;
            int old_idx = i * key_length + j;
            keys[new_idx] = keys[old_idx];
            if ((use_kca1 && random_num < prob_factors[i][j]) || (!use_kca1 && random_num > prob_factors[i][j])) {
                keys[new_idx] = 1 - keys[new_idx];
            }
            printf("%d ", keys[new_idx]);
        }
        printf("\n");
    }
}

// 🚀 Main KCA Optimization Function
void KCA_optimize(void* optimizer, ObjectiveFunction objective_function) {
    Optimizer* opt = (Optimizer*)optimizer;
    srand(42); // Fixed seed for reproducibility

    // Print bounds for debugging
    printf("Bounds: ");
    for (int i = 0; i < opt->dim * 2; i++) {
        printf("%f ", opt->bounds[i]);
    }
    printf("\n");

    // Calculate key length (bits per dimension * dimensions)
    int key_length = opt->dim * BITS_PER_DIM;
    int use_kca1 = 1; // Default to KCA1

    // Allocate temporary keys array
    int* keys = (int*)malloc(opt->population_size * key_length * sizeof(int));
    double* fitness = (double*)malloc(opt->population_size * sizeof(double));

    // Allocate probability factors
    int half_pop = opt->population_size / 2;
    double** prob_factors = (double**)malloc(half_pop * sizeof(double*));
    for (int i = 0; i < half_pop; i++) {
        prob_factors[i] = (double*)malloc(key_length * sizeof(double));
    }

    // Initialize keys
    kca_initialize_keys(opt, keys, key_length);

    // Main optimization loop
    for (int generation = 0; generation < opt->max_iter; generation++) {
        // Evaluate fitness
        kca_evaluate_keys(opt, keys, key_length, objective_function);

        // Find best solution
        int min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
        }
        if (opt->population[min_idx].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[min_idx].fitness;
            kca_binary_to_continuous(&keys[min_idx * key_length], opt->best_solution.position, key_length, opt->dim);
            printf("Updating best solution: fitness = %f, key = ", opt->best_solution.fitness);
            for (int j = 0; j < opt->dim; j++) {
                printf("%f ", opt->best_solution.position[j]);
            }
            printf("\n");
        }

        // Sort keys by fitness
        int* sorted_indices = (int*)malloc(opt->population_size * sizeof(int));
        for (int i = 0; i < opt->population_size; i++) {
            sorted_indices[i] = i;
            fitness[i] = opt->population[i].fitness;
        }
        printf("Before sorting:\n");
        for (int i = 0; i < opt->population_size; i++) {
            printf("Key %d fitness: %f\n", i, fitness[i]);
        }
        for (int i = 0; i < opt->population_size - 1; i++) {
            for (int j = 0; j < opt->population_size - i - 1; j++) {
                if (fitness[sorted_indices[j]] > fitness[sorted_indices[j + 1]]) {
                    int temp = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[j + 1];
                    sorted_indices[j + 1] = temp;
                }
            }
        }
        printf("After sorting:\n");
        for (int i = 0; i < opt->population_size; i++) {
            printf("Key %d fitness: %f\n", sorted_indices[i], fitness[sorted_indices[i]]);
        }
        int* temp_keys = (int*)malloc(opt->population_size * key_length * sizeof(int));
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < key_length; j++) {
                temp_keys[i * key_length + j] = keys[sorted_indices[i] * key_length + j];
            }
            kca_binary_to_continuous(&temp_keys[i * key_length], opt->population[i].position, key_length, opt->dim);
            opt->population[i].fitness = fitness[sorted_indices[i]];
        }
        memcpy(keys, temp_keys, opt->population_size * key_length * sizeof(int));
        free(temp_keys);
        free(sorted_indices);

        // Calculate probability factors
        kca_calculate_probability_factor(opt, keys, key_length, prob_factors);

        // Generate new keys
        kca_generate_new_keys(opt, keys, key_length, prob_factors, use_kca1);

        // Update population positions
        for (int i = 0; i < opt->population_size; i++) {
            kca_binary_to_continuous(&keys[i * key_length], opt->population[i].position, key_length, opt->dim);
        }

        // Enforce bounds
        enforce_bound_constraints(opt);

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", generation + 1, opt->best_solution.fitness);
    }

    // Free allocated memory
    free(keys);
    free(fitness);
    for (int i = 0; i < half_pop; i++) {
        free(prob_factors[i]);
    }
    free(prob_factors);
}