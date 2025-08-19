#include "DEA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h> // For SSE/AVX intrinsics
#include <stdint.h>

// Precomputed weights for fitness distribution (-10 to 10)
static const float WEIGHTS[21] __attribute__((aligned(64))) = {
    0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
    0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f
};

// Fast Xorshift RNG
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand_float(uint32_t *state) {
    return (float)(xorshift32(state) & 0xFFFFFF) / 0x1000000;
}

static inline float rand_float_range(float min, float max, uint32_t *state) {
    return min + (max - min) * rand_float(state);
}

// Initialize dolphin locations and alternatives
void initialize_locations(Optimizer *opt, DEAData *dea_data) {
    const int dim = opt->dim;
    dea_data->dim = dim;
    dea_data->pop_size = opt->population_size;
    dea_data->rng_state = (uint32_t)time(NULL);
    dea_data->dim_data = (DimensionData *)aligned_alloc(64, dim * sizeof(DimensionData));
    
    // Calculate effective radius
    float min_search_size = INFINITY;
    for (int j = 0; j < dim; j++) {
        float size = fabsf((float)(opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
        if (size < min_search_size) min_search_size = size;
    }
    dea_data->effective_radius = DEA_EFFECTIVE_RADIUS_FACTOR * min_search_size;

    // Initialize alternatives per dimension
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        dim->size = DEA_ALTERNATIVES_PER_DIM;
        dim->values = (float *)aligned_alloc(64, dim->size * sizeof(float));
        dim->accum_fitness = (float *)aligned_alloc(64, dim->size * sizeof(float));
        dim->probabilities = (float *)aligned_alloc(64, dim->size * sizeof(float));
        
        float lower = (float)opt->bounds[2 * j];
        float upper = (float)opt->bounds[2 * j + 1];
        float step = (upper - lower) / (dim->size - 1);
        float *values = dim->values;
        for (int k = 0; k < dim->size; k += 4) {
            values[k] = lower + k * step;
            if (k + 1 < dim->size) values[k + 1] = lower + (k + 1) * step;
            if (k + 2 < dim->size) values[k + 2] = lower + (k + 2) * step;
            if (k + 3 < dim->size) values[k + 3] = lower + (k + 3) * step;
            dim->accum_fitness[k] = 0.0f;
            dim->probabilities[k] = 0.0f;
        }
    }

    // Initialize random population
    for (int i = 0; i < opt->population_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            position[j] = (double)rand_float_range((float)opt->bounds[2 * j], (float)opt->bounds[2 * j + 1], &dea_data->rng_state);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Binary search for closest alternative
static inline int find_closest_alternative(const float *values, int size, float target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (values[mid] == target) return mid;
        if (values[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    if (left == 0) return 0;
    if (left == size) return size - 1;
    return (fabsf(values[left - 1] - target) < fabsf(values[left] - target)) ? left - 1 : left;
}

// Calculate accumulative fitness with SIMD
void calculate_accumulative_fitness(Optimizer *opt, float *fitness, DEAData *dea_data) {
    const int dim = dea_data->dim;
    const int pop_size = dea_data->pop_size;

    // Reset accumulative fitness with SIMD
    for (int j = 0; j < dim; j++) {
        float *accum_fitness = dea_data->dim_data[j].accum_fitness;
        const int size = dea_data->dim_data[j].size;
        int k = 0;
        __m256 zero = _mm256_setzero_ps();
        for (; k <= size - 8; k += 8) {
            _mm256_store_ps(accum_fitness + k, zero);
        }
        for (; k < size; k++) {
            accum_fitness[k] = 0.0f;
        }
    }

    // Distribute fitness
    for (int i = 0; i < pop_size; i++) {
        float fit = fitness[i];
        for (int j = 0; j < dim; j++) {
            DimensionData *dim = dea_data->dim_data + j;
            float loc_value = (float)opt->population[i].position[j];
            int alt_idx = find_closest_alternative(dim->values, dim->size, loc_value);
            float *accum_fitness = dim->accum_fitness;

            // Unrolled loop for weights
            for (int k = -10; k <= 10; k++) {
                int idx = alt_idx + k;
                if (idx >= 0 && idx < dim->size) {
                    accum_fitness[idx] += WEIGHTS[k + 10] * fit;
                }
            }
        }
    }
}

// Compute convergence probability
static inline float get_convergence_probability(int loop, int max_loops) {
    return powf((float)loop / max_loops, DEA_CONVERGENCE_POWER);
}

// Update probabilities with SIMD
void update_probabilities(Optimizer *opt, int loop, float *fitness, DEAData *dea_data) {
    const int dim = dea_data->dim;
    float convergence_prob = get_convergence_probability(loop, opt->max_iter);

    // Find best alternative indices
    int best_alt_indices[DEA_ALTERNATIVES_PER_DIM]; // Stack-allocated
    for (int j = 0; j < dim; j++) {
        best_alt_indices[j] = find_closest_alternative(dea_data->dim_data[j].values, dea_data->dim_data[j].size, (float)opt->best_solution.position[j]);
    }

    // Assign probabilities
    for (int j = 0; j < dim; j++) {
        DimensionData *dim = dea_data->dim_data + j;
        float *accum_fitness = dim->accum_fitness;
        float *probs = dim->probabilities;
        const int size = dim->size;
        float total_af = 0.0f;

        // SIMD sum for total accumulative fitness
        __m256 sum = _mm256_setzero_ps();
        int k = 0;
        for (; k <= size - 8; k += 8) {
            sum = _mm256_add_ps(sum, _mm256_load_ps(accum_fitness + k));
        }
        float sum_array[8];
        _mm256_store_ps(sum_array, sum);
        for (int i = 0; i < 8; i++) total_af += sum_array[i];
        for (; k < size; k++) {
            total_af += accum_fitness[k];
        }

        if (total_af == 0.0f) {
            float uniform_prob = 1.0f / size;
            __m256 uniform = _mm256_set1_ps(uniform_prob);
            for (k = 0; k <= size - 8; k += 8) {
                _mm256_store_ps(probs + k, uniform);
            }
            for (; k < size; k++) {
                probs[k] = uniform_prob;
            }
        } else {
            int best_idx = best_alt_indices[j];
            probs[best_idx] = convergence_prob;
            float remaining_prob = 1.0f - convergence_prob;
            float prob_sum = convergence_prob;

            // Compute and normalize probabilities
            __m256 total_af_vec = _mm256_set1_ps(total_af);
            __m256 remain_prob_vec = _mm256_set1_ps(remaining_prob);
            __m256 thresh_vec = _mm256_set1_ps(DEA_PROBABILITY_THRESHOLD);
            for (k = 0; k <= size - 8; k += 8) {
                if (k <= best_idx && best_idx < k + 8) continue;
                __m256 af = _mm256_load_ps(accum_fitness + k);
                __m256 prob = _mm256_mul_ps(_mm256_div_ps(af, total_af_vec), remain_prob_vec);
                prob = _mm256_max_ps(prob, thresh_vec);
                _mm256_store_ps(probs + k, prob);
            }
            for (k = 0; k < size; k++) {
                if (k != best_idx) {
                    float prob = (accum_fitness[k] / total_af) * remaining_prob;
                    probs[k] = prob > DEA_PROBABILITY_THRESHOLD ? prob : DEA_PROBABILITY_THRESHOLD;
                }
                prob_sum += probs[k];
            }

            // Normalize with SIMD
            if (prob_sum > 0.0f) {
                __m256 inv_sum = _mm256_set1_ps(1.0f / prob_sum);
                for (k = 0; k <= size - 8; k += 8) {
                    __m256 p = _mm256_load_ps(probs + k);
                    _mm256_store_ps(probs + k, _mm256_mul_ps(p, inv_sum));
                }
                for (; k < size; k++) {
                    probs[k] /= prob_sum;
                }
            }
        }
    }
}

// Select new locations
void select_new_locations(Optimizer *opt, DEAData *dea_data) {
    const int dim = dea_data->dim;
    const int pop_size = dea_data->pop_size;

    for (int i = 0; i < pop_size; i++) {
        double *position = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            DimensionData *dim = dea_data->dim_data + j;
            float *probs = dim->probabilities;
            float r = rand_float(&dea_data->rng_state);
            float cum_prob = 0.0f;
            int selected_idx = 0;

            // Unrolled loop for cumulative probability
            for (int k = 0; k < dim->size; k += 4) {
                if (r <= (cum_prob += probs[k])) { selected_idx = k; break; }
                if (k + 1 < dim->size && r <= (cum_prob += probs[k + 1])) { selected_idx = k + 1; break; }
                if (k + 2 < dim->size && r <= (cum_prob += probs[k + 2])) { selected_idx = k + 2; break; }
                if (k + 3 < dim->size && r <= (cum_prob += probs[k + 3])) { selected_idx = k + 3; break; }
            }
            position[j] = (double)dim->values[selected_idx];
        }
    }
    enforce_bound_constraints(opt);
}

// Main DEA optimization function
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    float *fitness = (float *)aligned_alloc(64, pop_size * sizeof(float));
    float prev_best_fitness = INFINITY;

    // Initialize DEA data
    DEAData dea_data;
    initialize_locations(opt, &dea_data);
    
    for (int loop = 0; loop < opt->max_iter; loop++) {
        // Evaluate fitness
        for (int i = 0; i < pop_size; i++) {
            fitness[i] = (float)objective_function(opt->population[i].position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                memcpy(opt->best_solution.position, opt->population[i].position, dim * sizeof(double));
            }
        }

        // Update probabilities and select new locations
        calculate_accumulative_fitness(opt, fitness, &dea_data);
        update_probabilities(opt, loop, fitness, &dea_data);
        select_new_locations(opt, &dea_data);

        // Check convergence
        if (loop > 0 && fabsf(opt->best_solution.fitness - prev_best_fitness) < 1e-6f) {
            printf("Convergence reached at loop %d.\n", loop + 1);
            break;
        }
        prev_best_fitness = opt->best_solution.fitness;
        printf("Loop %d: Best Value = %f\n", loop + 1, opt->best_solution.fitness);
    }

    // Cleanup
    for (int j = 0; j < dim; j++) {
        free(dea_data.dim_data[j].values);
        free(dea_data.dim_data[j].accum_fitness);
        free(dea_data.dim_data[j].probabilities);
    }
    free(dea_data.dim_data);
    free(fitness);
}
