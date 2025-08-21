#include "AMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>

// Fast Xorshift PRNG
static uint64_t xorshift_state = 1;
static inline double xorshift_rand() {
    xorshift_state ^= xorshift_state >> 12;
    xorshift_state ^= xorshift_state << 25;
    xorshift_state ^= xorshift_state >> 27;
    return ((xorshift_state * 0x2545F4914F6CDD1DULL) >> 32) / 4294967296.0;
}

static inline double rand_double(double min, double max) {
    return min + (max - min) * xorshift_rand();
}

// Precomputed Gaussian variates
static double normal_table[NORMAL_TABLE_SIZE];
static int normal_table_initialized = 0;

static inline double normal_rand(double mean, double stddev) {
    if (!normal_table_initialized) {
        for (int i = 0; i < NORMAL_TABLE_SIZE; i++) {
            double u, v, s;
            do {
                u = rand_double(-1.0, 1.0);
                v = rand_double(-1.0, 1.0);
                s = u * u + v * v;
            } while (s >= 1.0 || s == 0.0);
            s = sqrt(-2.0 * log(s) / s);
            normal_table[i] = mean + stddev * u * s;
        }
        normal_table_initialized = 1;
    }
    return normal_table[rand() % NORMAL_TABLE_SIZE];
}

// Radix sort for floating-point fitness values
static void radix_sort_indices(Optimizer *opt, int *indices, double *fitness, int n) {
    const int BUCKETS = 256;
    int *temp_indices = (int*)ALIGNED_MALLOC(n * sizeof(int));
    int counts[BUCKETS];
    
    // Process each byte of the double
    for (int byte = 0; byte < 8; byte++) {
        memset(counts, 0, BUCKETS * sizeof(int));
        
        // Count occurrences
        for (int i = 0; i < n; i++) {
            uint64_t val = *(uint64_t*)&fitness[indices[i]];
            counts[(val >> (byte * 8)) & 0xFF]++;
        }
        
        // Cumulative counts
        for (int i = 1; i < BUCKETS; i++) {
            counts[i] += counts[i - 1];
        }
        
        // Distribute indices
        for (int i = n - 1; i >= 0; i--) {
            uint64_t val = *(uint64_t*)&fitness[indices[i]];
            temp_indices[--counts[(val >> (byte * 8)) & 0xFF]] = indices[i];
        }
        
        // Copy back
        memcpy(indices, temp_indices, n * sizeof(int));
    }
    
    // Handle negative numbers
    int start = 0, end = n - 1;
    while (start < end) {
        while (start < n && fitness[indices[start]] >= 0) start++;
        while (end >= 0 && fitness[indices[end]] < 0) end--;
        if (start < end) {
            int temp = indices[start];
            indices[start] = indices[end];
            indices[end] = temp;
        }
    }
    
    ALIGNED_FREE(temp_indices);
}

// Initialize Population
void initialize_population_amo(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < FIXED_DIM; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Neighborhood Learning Phase with SIMD
void neighborhood_learning_phase(Optimizer *opt) {
    double *new_pop = (double*)ALIGNED_MALLOC(opt->population_size * FIXED_DIM * sizeof(double));
    
    for (int i = 0; i < opt->population_size; i++) {
        double FF = normal_rand(0.0, 1.0);
        int lseq[NEIGHBORHOOD_SIZE];
        
        // Define neighborhood
        lseq[0] = (i == 0) ? opt->population_size - 2 : (i == 1) ? opt->population_size - 1 : (i == opt->population_size - 2) ? i - 2 : (i == opt->population_size - 1) ? i - 2 : i - 2;
        lseq[1] = (i == 0) ? opt->population_size - 1 : (i == 1) ? i - 1 : (i == opt->population_size - 2) ? i - 1 : (i == opt->population_size - 1) ? i - 1 : i - 1;
        lseq[2] = i;
        lseq[3] = (i == 0) ? i + 1 : (i == 1) ? i + 1 : (i == opt->population_size - 2) ? opt->population_size - 1 : (i == opt->population_size - 1) ? 0 : i + 1;
        lseq[4] = (i == 0) ? i + 2 : (i == 1) ? i + 2 : (i == opt->population_size - 2) ? 0 : (i == opt->population_size - 1) ? 1 : i + 2;
        
        // Random permutation
        for (int j = 0; j < NEIGHBORHOOD_SIZE; j++) {
            int temp = lseq[j];
            int idx = rand() % NEIGHBORHOOD_SIZE;
            lseq[j] = lseq[idx];
            lseq[idx] = temp;
        }
        
        int exemplar_idx = lseq[1];
        int offset = i * FIXED_DIM;
        double *curr_pos = opt->population[i].position;
        double *exem_pos = opt->population[exemplar_idx].position;
        
        // SIMD update for dim=2
        __m128d curr = _mm_loadu_pd(curr_pos);
        __m128d exem = _mm_loadu_pd(exem_pos);
        __m128d diff = _mm_sub_pd(exem, curr);
        __m128d ff = _mm_set1_pd(FF);
        __m128d update = _mm_mul_pd(ff, diff);
        __m128d new_pos = _mm_add_pd(curr, update);
        _mm_storeu_pd(&new_pop[offset], new_pos);
    }
    
    // Copy back and enforce bounds
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * FIXED_DIM;
        double *pos = opt->population[i].position;
        __m128d new_pos = _mm_loadu_pd(&new_pop[offset]);
        __m128d bound_min = _mm_loadu_pd(&opt->bounds[0]);
        __m128d bound_max = _mm_loadu_pd(&opt->bounds[2]);
        new_pos = _mm_max_pd(new_pos, bound_min);
        new_pos = _mm_min_pd(new_pos, bound_max);
        _mm_storeu_pd(pos, new_pos);
    }
    
    ALIGNED_FREE(new_pop);
}

// Global Migration Phase
void global_migration_phase(Optimizer *opt) {
    double *new_pop = (double*)ALIGNED_MALLOC(opt->population_size * FIXED_DIM * sizeof(double));
    double *probabilities = (double*)ALIGNED_MALLOC(opt->population_size * sizeof(double));
    int *sort_indices = (int*)ALIGNED_MALLOC(opt->population_size * sizeof(int));
    int *r1 = (int*)ALIGNED_MALLOC(opt->population_size * sizeof(int));
    int *r3 = (int*)ALIGNED_MALLOC(opt->population_size * sizeof(int));
    double *fitness = (double*)ALIGNED_MALLOC(opt->population_size * sizeof(double));
    
    // Collect fitness values
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
        sort_indices[i] = i;
    }
    
    // Sort indices by fitness
    radix_sort_indices(opt, sort_indices, fitness, opt->population_size);
    
    // Assign probabilities
    for (int i = 0; i < opt->population_size; i++) {
        probabilities[sort_indices[i]] = (opt->population_size - i) / (double)opt->population_size;
    }
    
    // Generate random indices
    for (int i = 0; i < opt->population_size; i++) {
        int indices[opt->population_size];
        for (int j = 0; j < opt->population_size; j++) {
            indices[j] = j;
        }
        
        // Fisher-Yates shuffle
        for (int j = opt->population_size - 1; j > 0; j--) {
            int k = (int)(xorshift_rand() * (j + 1));
            int temp = indices[j];
            indices[j] = indices[k];
            indices[k] = temp;
        }
        
        int idx = 0;
        while (indices[idx] == i) idx++;
        r1[i] = indices[idx++];
        while (indices[idx] == i) idx++;
        r3[i] = indices[idx];
    }
    
    // Update population with SIMD
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * FIXED_DIM;
        double *curr_pos = opt->population[i].position;
        double *r1_pos = opt->population[r1[i]].position;
        double *r3_pos = opt->population[r3[i]].position;
        double *best_pos = opt->best_solution.position;
        
        if (xorshift_rand() > probabilities[i]) {
            __m128d curr = _mm_loadu_pd(curr_pos);
            __m128d r1_vec = _mm_loadu_pd(r1_pos);
            __m128d r3_vec = _mm_loadu_pd(r3_pos);
            __m128d best = _mm_loadu_pd(best_pos);
            __m128d rand1 = _mm_set1_pd(rand_double(0.0, 1.0));
            __m128d rand2 = _mm_set1_pd(rand_double(0.0, 1.0));
            __m128d diff1 = _mm_sub_pd(best, curr);
            __m128d diff2 = _mm_sub_pd(r3_vec, curr);
            __m128d term1 = _mm_mul_pd(rand1, diff1);
            __m128d term2 = _mm_mul_pd(rand2, diff2);
            __m128d new_pos = _mm_add_pd(r1_vec, _mm_add_pd(term1, term2));
            _mm_storeu_pd(&new_pop[offset], new_pos);
        } else {
            __m128d curr = _mm_loadu_pd(curr_pos);
            _mm_storeu_pd(&new_pop[offset], curr);
        }
    }
    
    // Copy back and enforce bounds
    for (int i = 0; i < opt->population_size; i++) {
        int offset = i * FIXED_DIM;
        double *pos = opt->population[i].position;
        __m128d new_pos = _mm_loadu_pd(&new_pop[offset]);
        __m128d bound_min = _mm_loadu_pd(&opt->bounds[0]);
        __m128d bound_max = _mm_loadu_pd(&opt->bounds[2]);
        new_pos = _mm_max_pd(new_pos, bound_min);
        new_pos = _mm_min_pd(new_pos, bound_max);
        _mm_storeu_pd(pos, new_pos);
    }
    
    ALIGNED_FREE(new_pop);
    ALIGNED_FREE(probabilities);
    ALIGNED_FREE(sort_indices);
    ALIGNED_FREE(r1);
    ALIGNED_FREE(r3);
    ALIGNED_FREE(fitness);
}

// Main Optimization Function
void AMO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer*)opt_void;
    xorshift_state = (uint64_t)time(NULL);
    
    initialize_population_amo(opt);
    
    // Batch fitness evaluation
    char *needs_eval = (char*)calloc(opt->population_size, sizeof(char));
    for (int i = 0; i < opt->population_size; i++) {
        needs_eval[i] = 1;
    }
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness only for marked individuals
        for (int i = 0; i < opt->population_size; i++) {
            if (needs_eval[i]) {
                double new_fitness = objective_function(opt->population[i].position);
                if (new_fitness <= opt->population[i].fitness) {
                    opt->population[i].fitness = new_fitness;
                    if (new_fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = new_fitness;
                        for (int j = 0; j < FIXED_DIM; j++) {
                            opt->best_solution.position[j] = opt->population[i].position[j];
                        }
                    }
                }
            }
        }
        
        // Neighborhood learning phase
        neighborhood_learning_phase(opt);
        memset(needs_eval, 1, opt->population_size * sizeof(char));
        
        // Global migration phase
        global_migration_phase(opt);
        memset(needs_eval, 1, opt->population_size * sizeof(char));
        
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    free(needs_eval);
}
