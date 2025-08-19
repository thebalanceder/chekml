#include "AAA.h"
#include "generaloptimizer.h"
#include <emmintrin.h> // SSE2
#include <immintrin.h> // AVX2
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Assume 32-byte alignment for AVX2
#define ALIGNMENT 32

// Fast Xorshift RNG
static unsigned int xorshift_state = 1;
static void init_xorshift() {
    xorshift_state = (unsigned int)time(NULL);
}

static inline unsigned int xorshift32() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static inline double fast_rand_double(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / 0xffffffffU);
}

// Aligned memory allocation
static inline double* alloc_aligned(int size) {
    void *ptr;
    if (posix_memalign(&ptr, ALIGNMENT, size * sizeof(double)) != 0) {
        return NULL;
    }
    return (double*)ptr;
}

// Initialize Population with SIMD
static inline void initialize_population(Optimizer *opt) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    double *bounds = opt->bounds;
    
    // Precompute bound differences
    double *bound_diffs = alloc_aligned(dim);
    if (!bound_diffs) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    for (int j = 0; j < dim; j++) {
        bound_diffs[j] = bounds[2 * j + 1] - bounds[2 * j];
    }
    
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int j = 0;
        // SIMD for 4 doubles at a time (AVX2)
        for (; j <= dim - 4; j += 4) {
            __m256d rand = _mm256_set1_pd(fast_rand_double(0.0, 1.0));
            __m256d lower = _mm256_loadu_pd(&bounds[2 * j]);
            __m256d diff = _mm256_load_pd(&bound_diffs[j]);
            __m256d result = _mm256_add_pd(lower, _mm256_mul_pd(rand, diff));
            _mm256_store_pd(&pos[j], result);
        }
        // Handle remaining dimensions
        for (; j < dim; j++) {
            pos[j] = bounds[2 * j] + fast_rand_double(0.0, 1.0) * bound_diffs[j];
        }
        opt->population[i].fitness = INFINITY;
    }
    free(bound_diffs);
    enforce_bound_constraints(opt);
}

// Evaluate Population
static inline void evaluate_population(Optimizer *opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    // Unroll loop for small pop_size
    int i = 0;
    for (; i <= pop_size - 4; i += 4) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        opt->population[i + 1].fitness = objective_function(opt->population[i + 1].position);
        opt->population[i + 2].fitness = objective_function(opt->population[i + 2].position);
        opt->population[i + 3].fitness = objective_function(opt->population[i + 3].position);
    }
    for (; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Movement Phase with SIMD and Inline Bounds
static inline void movement_phase(Optimizer *opt) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double *best_pos = opt->best_solution.position;
    const double *bounds = opt->bounds;
    const __m256d step_vec = _mm256_set1_pd(STEP_SIZE);
    
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int j = 0;
        // Process 4 dimensions at a time with AVX2
        for (; j <= dim - 4; j += 4) {
            __m256d pos_vec = _mm256_load_pd(&pos[j]);
            __m256d best_vec = _mm256_loadu_pd(&best_pos[j]);
            __m256d direction = _mm256_sub_pd(best_vec, pos_vec);
            __m256d update = _mm256_mul_pd(step_vec, direction);
            __m256d new_pos = _mm256_add_pd(pos_vec, update);
            
            // Inline bounds checking
            __m256d lower = _mm256_loadu_pd(&bounds[2 * j]);
            __m256d upper = _mm256_loadu_pd(&bounds[2 * j + 1]);
            new_pos = _mm256_max_pd(new_pos, lower);
            new_pos = _mm256_min_pd(new_pos, upper);
            _mm256_store_pd(&pos[j], new_pos);
        }
        // Handle remaining dimensions
        for (; j < dim; j++) {
            double direction = best_pos[j] - pos[j];
            pos[j] += STEP_SIZE * direction;
            if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
            else if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
        }
    }
}

// Main Optimization Function
void AAA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    init_xorshift();
    
    // Align best_solution.position
    double *aligned_best = alloc_aligned(opt->dim);
    if (!aligned_best) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    memcpy(aligned_best, opt->best_solution.position, opt->dim * sizeof(double));
    opt->best_solution.position = aligned_best;
    
    double *temp_pos = alloc_aligned(opt->dim);
    if (!temp_pos) {
        fprintf(stderr, "Memory allocation failed\n");
        free(aligned_best);
        return;
    }
    
    initialize_population(opt);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        double best_fitness = opt->best_solution.fitness;
        const int pop_size = opt->population_size;
        const int dim = opt->dim;
        int best_idx = -1;
        
        // Evaluate and find best with unrolled loop
        int i = 0;
        for (; i <= pop_size - 4; i += 4) {
            double f0 = objective_function(opt->population[i].position);
            double f1 = objective_function(opt->population[i + 1].position);
            double f2 = objective_function(opt->population[i + 2].position);
            double f3 = objective_function(opt->population[i + 3].position);
            opt->population[i].fitness = f0;
            opt->population[i + 1].fitness = f1;
            opt->population[i + 2].fitness = f2;
            opt->population[i + 3].fitness = f3;
            if (f0 < best_fitness) { best_fitness = f0; best_idx = i; }
            if (f1 < best_fitness) { best_fitness = f1; best_idx = i + 1; }
            if (f2 < best_fitness) { best_fitness = f2; best_idx = i + 2; }
            if (f3 < best_fitness) { best_fitness = f3; best_idx = i + 3; }
        }
        for (; i < pop_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < best_fitness) { best_fitness = fitness; best_idx = i; }
        }
        
        // Update best solution
        if (best_idx >= 0) {
            opt->best_solution.fitness = best_fitness;
            memcpy(aligned_best, opt->population[best_idx].position, dim * sizeof(double));
        }
        
        movement_phase(opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    free(temp_pos);
    free(aligned_best);
}
