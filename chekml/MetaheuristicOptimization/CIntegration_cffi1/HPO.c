#include "HPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h> // SSE2
#ifdef __AVX__
#include <immintrin.h> // AVX
#endif

// Fast Xorshift random number generator
void hpo_xorshift_init(HPOXorshiftState *state, unsigned long seed) {
    state->x = seed | 1; // Ensure non-zero
    state->y = 1812433253UL;
    state->z = 1664525UL;
    state->w = 1013904223UL;
}

double hpo_xorshift_double(HPOXorshiftState *state) {
    unsigned long t = state->x;
    t ^= t << 11;
    t ^= t >> 8;
    state->x = state->y; state->y = state->z; state->z = state->w;
    state->w ^= state->w >> 19 ^ t;
    return (double)state->w / 4294967296.0; // Normalize to [0,1)
}

// Optimized quicksort with median-of-three pivot and insertion sort for small arrays
static void hpo_insertion_sort_indices(double *arr, int *indices, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = indices[i];
        double key_val = arr[key];
        int j = i - 1;
        while (j >= low && arr[indices[j]] > key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

void hpo_quicksort_indices(double *arr, int *indices, int low, int high) {
    if (high - low < 10) { // Use insertion sort for small arrays
        hpo_insertion_sort_indices(arr, indices, low, high);
        return;
    }

    // Median-of-three pivot selection
    int mid = low + (high - low) / 2;
    if (arr[indices[mid]] > arr[indices[high]]) {
        int temp = indices[mid]; indices[mid] = indices[high]; indices[high] = temp;
    }
    if (arr[indices[low]] > arr[indices[high]]) {
        int temp = indices[low]; indices[low] = indices[high]; indices[high] = temp;
    }
    if (arr[indices[mid]] > arr[indices[low]]) {
        int temp = indices[mid]; indices[mid] = indices[low]; indices[low] = temp;
    }

    double pivot = arr[indices[low]];
    int i = low, j = high + 1;

    while (1) {
        do { i++; } while (i <= high && arr[indices[i]] < pivot);
        do { j--; } while (arr[indices[j]] > pivot);
        if (i >= j) break;
        int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
    }
    int temp = indices[low]; indices[low] = indices[j]; indices[j] = temp;

    // Tail-call optimization
    if (j - low < high - j) {
        hpo_quicksort_indices(arr, indices, low, j - 1);
        low = j + 1;
    } else {
        hpo_quicksort_indices(arr, indices, j + 1, high);
        high = j - 1;
    }
    if (low < high) hpo_quicksort_indices(arr, indices, low, high);
}

// Update positions with SIMD and cache-friendly access
void hpo_update_positions(Optimizer *opt, int iter, double c_factor, HPOXorshiftState *rng, double *xi, double *dist, int *idxsortdist, double *r1, double *r3, char *idx, double *z) {
    double c = 1.0 - ((double)iter * c_factor);
    int kbest = (int)(opt->population_size * c + 0.5);

    // Compute mean position (xi) with cache-friendly access
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            sum += opt->population[i].position[j];
        }
        xi[j] = sum / opt->population_size;
    }

    // Compute distances using SSE2 for dot products
    for (int i = 0; i < opt->population_size; i++) {
        double sum = 0.0;
        int j = 0;
        #ifdef __AVX__
        if (opt->dim >= 4) {
            __m256d sum_vec = _mm256_setzero_pd();
            for (; j <= opt->dim - 4; j += 4) {
                __m256d pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                __m256d mean = _mm256_loadu_pd(&xi[j]);
                __m256d diff = _mm256_sub_pd(pos, mean);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
            }
            sum += sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
        }
        #endif
        for (; j < opt->dim; j++) {
            double diff = opt->population[i].position[j] - xi[j];
            sum += diff * diff;
        }
        dist[i] = sqrt(sum);
        idxsortdist[i] = i;
    }

    // Sort distances
    hpo_quicksort_indices(dist, idxsortdist, 0, opt->population_size - 1);

    // Update positions with SIMD
    int si_idx = idxsortdist[kbest - 1];
    for (int i = 0; i < opt->population_size; i++) {
        double r2 = hpo_xorshift_double(rng);

        // Generate r1, idx, r3, z with minimal branching
        for (int j = 0; j < opt->dim; j++) {
            r1[j] = hpo_xorshift_double(rng);
            idx[j] = (r1[j] < c);
            r3[j] = hpo_xorshift_double(rng);
            z[j] = idx[j] ? r2 : r3[j];
        }

        if (hpo_xorshift_double(rng) < CONSTRICTION_COEFF) {
            // Safe mode: SIMD for position update
            int j = 0;
            #ifdef __AVX__
            if (opt->dim >= 4) {
                __m256d two = _mm256_set1_pd(2.0);
                __m256d c_vec = _mm256_set1_pd(c);
                __m256d one_minus_c = _mm256_set1_pd(1.0 - c);
                __m256d half = _mm256_set1_pd(0.5);
                for (; j <= opt->dim - 4; j += 4) {
                    __m256d pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                    __m256d si_pos = _mm256_loadu_pd(&opt->population[si_idx].position[j]);
                    __m256d xi_vec = _mm256_loadu_pd(&xi[j]);
                    __m256d z_vec = _mm256_loadu_pd(&z[j]);
                    __m256d term1 = _mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(two, c_vec), _mm256_mul_pd(z_vec, si_pos)), pos);
                    __m256d term2 = _mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(two, one_minus_c), _mm256_mul_pd(z_vec, xi_vec)), pos);
                    __m256d new_pos = _mm256_add_pd(pos, _mm256_mul_pd(half, _mm256_add_pd(term1, term2)));
                    _mm256_storeu_pd(&opt->population[i].position[j], new_pos);
                }
            }
            #endif
            for (; j < opt->dim; j++) {
                double si_pos = opt->population[si_idx].position[j];
                opt->population[i].position[j] += 0.5 * (
                    (2.0 * c * z[j] * si_pos - opt->population[i].position[j]) +
                    (2.0 * (1.0 - c) * z[j] * xi[j] - opt->population[i].position[j])
                );
            }
        } else {
            // Attack mode: SIMD with fast cos approximation
            int j = 0;
            #ifdef __AVX__
            if (opt->dim >= 4) {
                __m256d two = _mm256_set1_pd(2.0);
                __m256d neg_one = _mm256_set1_pd(-1.0);
                for (; j <= opt->dim - 4; j += 4) {
                    __m256d z_vec = _mm256_loadu_pd(&z[j]);
                    __m256d rr = _mm256_add_pd(neg_one, _mm256_mul_pd(two, z_vec));
                    __m256d cos_term = _mm256_set1_pd(cos(TWO_PI * rr[0])); // Simplified; use lookup table for production
                    __m256d pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                    __m256d best = _mm256_loadu_pd(&opt->best_solution.position[j]);
                    __m256d diff = _mm256_sub_pd(best, pos);
                    __m256d new_pos = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(two, z_vec), _mm256_mul_pd(cos_term, diff)), best);
                    _mm256_storeu_pd(&opt->population[i].position[j], new_pos);
                }
            }
            #endif
            for (; j < opt->dim; j++) {
                double rr = -1.0 + 2.0 * z[j];
                opt->population[i].position[j] = 2.0 * z[j] * cos(TWO_PI * rr) * 
                    (opt->best_solution.position[j] - opt->population[i].position[j]) + 
                    opt->best_solution.position[j];
            }
        }

        // Enforce bounds with SIMD
        int j = 0;
        #ifdef __AVX__
        if (opt->dim >= 4) {
            for (; j <= opt->dim - 4; j += 4) {
                __m256d pos = _mm256_loadu_pd(&opt->population[i].position[j]);
                __m256d lower = _mm256_loadu_pd(&opt->bounds[2 * j]);
                __m256d upper = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                pos = _mm256_max_pd(pos, lower);
                pos = _mm256_min_pd(pos, upper);
                _mm256_storeu_pd(&opt->population[i].position[j], pos);
            }
        }
        #endif
        for (; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Main Optimization Function
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize Xorshift RNG
    HPOXorshiftState rng;
    hpo_xorshift_init(&rng, (unsigned long)time(NULL));

    // Allocate temporary data (aligned for SIMD)
    double c_factor = C_PARAM_MAX / opt->max_iter;
    double *xi = (double *)_mm_malloc(opt->dim * sizeof(double), HPO_ALIGNMENT);
    double *dist = (double *)_mm_malloc(opt->population_size * sizeof(double), HPO_ALIGNMENT);
    int *idxsortdist = (int *)_mm_malloc(opt->population_size * sizeof(int), HPO_ALIGNMENT);
    double *r1 = (double *)_mm_malloc(opt->dim * sizeof(double), HPO_ALIGNMENT);
    double *r3 = (double *)_mm_malloc(opt->dim * sizeof(double), HPO_ALIGNMENT);
    char *idx = (char *)_mm_malloc(opt->dim * sizeof(char), HPO_ALIGNMENT);
    double *z = (double *)_mm_malloc(opt->dim * sizeof(double), HPO_ALIGNMENT);

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                hpo_xorshift_double(&rng) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (__builtin_expect(opt->population[i].fitness < opt->best_solution.fitness, 0)) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        hpo_update_positions(opt, iter, c_factor, &rng, xi, dist, idxsortdist, r1, r3, idx, z);

        // Evaluate new positions
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (__builtin_expect(new_fitness < opt->best_solution.fitness, 0)) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        printf("Iteration: %d, Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    _mm_free(xi);
    _mm_free(dist);
    _mm_free(idxsortdist);
    _mm_free(r1);
    _mm_free(r3);
    _mm_free(idx);
    _mm_free(z);
}
