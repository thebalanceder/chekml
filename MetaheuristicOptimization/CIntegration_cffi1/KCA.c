#include "KCA.h"
#include <string.h>

// Precompute bit scales
double BIT_SCALES[32] = {0};

// Initialize BIT_SCALES
__attribute__((constructor)) void init_bit_scales() {
    for (int b = 0; b < 32; b++) {
        BIT_SCALES[b] = 1.0 / (1ULL << (b + 1));
    }
}

// 🌟 Fast random number generator (LCG)
uint32_t rng_state = 42; // Global state for reproducibility

static inline uint32_t fast_rand() {
    rng_state = rng_state * 1664525 + 1013904223; // LCG parameters
    return rng_state;
}

static inline double fast_rand_double(double min, double max) {
    // Use 32-bit random number scaled to [0, 1)
    uint32_t r = fast_rand();
    double scale = (double)r / 4294967296.0; // 2^32
    return min + scale * (max - min);
}

// 🌟 Optimized binary to continuous conversion
void kca_binary_to_continuous(const int* binary_key, double* continuous_key, int key_length, int dim) {
    int bits_per_dim = key_length / dim;
    for (int d = 0; d < dim; d++) {
        double value = 0.0;
        const int* key_ptr = binary_key + d * bits_per_dim;
        // Unroll inner loop for small bits_per_dim (common case: 10)
        for (int b = 0; b < bits_per_dim; b += 4) {
            if (b + 0 < bits_per_dim) value += key_ptr[b + 0] * BIT_SCALES[b + 0];
            if (b + 1 < bits_per_dim) value += key_ptr[b + 1] * BIT_SCALES[b + 1];
            if (b + 2 < bits_per_dim) value += key_ptr[b + 2] * BIT_SCALES[b + 2];
            if (b + 3 < bits_per_dim) value += key_ptr[b + 3] * BIT_SCALES[b + 3];
        }
        continuous_key[d] = SHUBERT_MIN + value * SHUBERT_RANGE;
    }
}

// 🌱 Initialize keys
void kca_initialize_keys(Optimizer* opt, int* keys, int key_length) {
    int total_bits = opt->population_size * key_length;
    // Vectorizable loop for random bit initialization
    for (int j = 0; j < total_bits; j++) {
        keys[j] = (fast_rand_double(0.0, 1.0) < KEY_INIT_PROB) ? 1 : 0;
    }
}

// 📊 Evaluate fitness
void kca_evaluate_keys(Optimizer* opt, int* keys, int key_length, double (*objective_function)(double*)) {
    double* temp_position = (double*)malloc(opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        kca_binary_to_continuous(keys + i * key_length, temp_position, key_length, opt->dim);
        opt->population[i].fitness = objective_function(temp_position) * FITNESS_SCALE;
    }
    free(temp_position);
}

// 🔍 Optimized probability factor calculation
void kca_calculate_probability_factor(Optimizer* opt, const int* keys, int key_length, double* prob_factors) {
    int half_pop = opt->population_size / 2;
    for (int j = 0; j < key_length; j++) {
        int tooth_sum = 0;
        // Unroll loop for better performance
        for (int i = 0; i < half_pop; i += 4) {
            if (i + 0 < half_pop) tooth_sum += keys[(i + 0) * key_length + j];
            if (i + 1 < half_pop) tooth_sum += keys[(i + 1) * key_length + j];
            if (i + 2 < half_pop) tooth_sum += keys[(i + 2) * key_length + j];
            if (i + 3 < half_pop) tooth_sum += keys[(i + 3) * key_length + j];
        }
        double prob = 1.0 - ((double)tooth_sum / half_pop);
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
        // Unroll loop for key generation
        for (int j = 0; j < key_length; j += 4) {
            if (j + 0 < key_length) {
                new_key[j + 0] = old_key[j + 0];
                if ((use_kca1 && fast_rand_double(0.0, 1.0) < prob[j + 0]) ||
                    (!use_kca1 && fast_rand_double(0.0, 1.0) > prob[j + 0])) {
                    new_key[j + 0] = 1 - new_key[j + 0];
                }
            }
            if (j + 1 < key_length) {
                new_key[j + 1] = old_key[j + 1];
                if ((use_kca1 && fast_rand_double(0.0, 1.0) < prob[j + 1]) ||
                    (!use_kca1 && fast_rand_double(0.0, 1.0) > prob[j + 1])) {
                    new_key[j + 1] = 1 - new_key[j + 1];
                }
            }
            if (j + 2 < key_length) {
                new_key[j + 2] = old_key[j + 2];
                if ((use_kca1 && fast_rand_double(0.0, 1.0) < prob[j + 2]) ||
                    (!use_kca1 && fast_rand_double(0.0, 1.0) > prob[j + 2])) {
                    new_key[j + 2] = 1 - new_key[j + 2];
                }
            }
            if (j + 3 < key_length) {
                new_key[j + 3] = old_key[j + 3];
                if ((use_kca1 && fast_rand_double(0.0, 1.0) < prob[j + 3]) ||
                    (!use_kca1 && fast_rand_double(0.0, 1.0) > prob[j + 3])) {
                    new_key[j + 3] = 1 - new_key[j + 3];
                }
            }
        }
    }
}

// 🌟 Comparison function for qsort
static double* fitness_for_compare = NULL;

static int compare_fitness(const void* a, const void* b) {
    double fa = fitness_for_compare[*(const int*)a];
    double fb = fitness_for_compare[*(const int*)b];
    return (fa > fb) - (fa < fb);
}

// 🚀 Main KCA Optimization Function
void KCA_optimize(void* optimizer, ObjectiveFunction objective_function) {
    Optimizer* opt = (Optimizer*)optimizer;
    rng_state = 42; // Reset RNG for reproducibility

    int key_length = opt->dim * BITS_PER_DIM;
    int use_kca1 = 1;
    int half_pop = opt->population_size / 2;

    // Allocate memory
    int* keys = (int*)malloc(opt->population_size * key_length * sizeof(int));
    double* prob_factors = (double*)malloc(half_pop * key_length * sizeof(double));
    double* fitness = (double*)malloc(opt->population_size * sizeof(double));
    int* sorted_indices = (int*)malloc(opt->population_size * sizeof(int));
    int* temp_keys = (int*)malloc(opt->population_size * key_length * sizeof(int));
    double* temp_position = (double*)malloc(opt->dim * sizeof(double)); // Reused across iterations

    // Initialize keys
    kca_initialize_keys(opt, keys, key_length);

    // Main optimization loop
    for (int generation = 0; generation < opt->max_iter; generation++) {
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            kca_binary_to_continuous(keys + i * key_length, temp_position, key_length, opt->dim);
            opt->population[i].fitness = objective_function(temp_position) * FITNESS_SCALE;
        }

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
        fitness_for_compare = fitness;
        qsort(sorted_indices, opt->population_size, sizeof(int), compare_fitness);
        fitness_for_compare = NULL;

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
    }

    // Free allocated memory
    free(temp_position);
    free(keys);
    free(prob_factors);
    free(fitness);
    free(sorted_indices);
    free(temp_keys);
}
