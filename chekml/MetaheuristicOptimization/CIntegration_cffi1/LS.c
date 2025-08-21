#include "LS.h"
#include "generaloptimizer.h"
#include <stdint.h>  // For uint64_t in Xorshift
#include <emmintrin.h>  // For SSE2 intrinsics
#include <time.h>  // For initial seed

// 🔧 Xorshift RNG for extreme speed
static uint64_t rng_state = 1;
static inline double fast_rand_double(double min, double max) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t r = rng_state * 0x2545F4914F6CDD1DULL;
    double x = (double)(r >> 32) / 4294967296.0;
    return min + (max - min) * x;
}

// 🚀 Main Local Search Optimization Function
void LS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Stack-based storage for small dimensions
    double current_solution[MAX_DIM] __attribute__((aligned(CACHE_LINE)));
    double neighbors[NEIGHBOR_COUNT * MAX_DIM] __attribute__((aligned(CACHE_LINE)));
    double current_value, best_neighbor_value;
    int iteration, i, j;

    // Validate dimensions
    if (opt->dim > MAX_DIM) {
        fprintf(stderr, "Dimension %d exceeds MAX_DIM %d\n", opt->dim, MAX_DIM);
        exit(1);
    }

    // Initialize RNG seed
    if (rng_state == 1) {
        rng_state = (uint64_t)time(NULL);
    }

    // Initialize random solution
    for (j = 0; j < opt->dim; j++) {
        current_solution[j] = fast_rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
    }
    current_value = objective_function(current_solution);

    // Set initial best solution
    opt->best_solution.fitness = current_value;
    for (j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = current_solution[j];
    }

    // Optimization loop
    #pragma GCC unroll 4
    for (iteration = 0; iteration < MAX_ITER; iteration++) {
        // Generate neighbors with SIMD
        __m128d step_size = _mm_set1_pd(STEP_SIZE);
        __m128d neg_step_size = _mm_set1_pd(-STEP_SIZE);
        for (i = 0; i < NEIGHBOR_COUNT; i++) {
            for (j = 0; j < opt->dim; j += 2) {
                if (j + 1 < opt->dim) {
                    // SIMD perturbation
                    __m128d rand = _mm_set_pd(fast_rand_double(0.0, 1.0), fast_rand_double(0.0, 1.0));
                    __m128d perturb = _mm_mul_pd(rand, _mm_sub_pd(step_size, neg_step_size));
                    perturb = _mm_add_pd(perturb, neg_step_size);
                    __m128d curr = _mm_load_pd(&current_solution[j]);  // Aligned load
                    __m128d new_pos = _mm_add_pd(curr, perturb);
                    // SIMD bounds clipping
                    __m128d lower = _mm_set_pd(opt->bounds[2 * (j + 1)], opt->bounds[2 * j]);
                    __m128d upper = _mm_set_pd(opt->bounds[2 * (j + 1) + 1], opt->bounds[2 * j + 1]);
                    new_pos = _mm_max_pd(new_pos, lower);
                    new_pos = _mm_min_pd(new_pos, upper);
                    _mm_store_pd(&neighbors[i * opt->dim + j], new_pos);  // Aligned store
                } else {
                    // Scalar for odd dimension
                    neighbors[i * opt->dim + j] = current_solution[j] + fast_rand_double(-STEP_SIZE, STEP_SIZE);
                    neighbors[i * opt->dim + j] = neighbors[i * opt->dim + j] < opt->bounds[2 * j] ? opt->bounds[2 * j] :
                                                  neighbors[i * opt->dim + j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] :
                                                  neighbors[i * opt->dim + j];
                }
            }
        }

        // Evaluate neighbors
        best_neighbor_value = current_value;
        int best_neighbor_idx = -1;

        #pragma GCC unroll 4
        for (i = 0; i < NEIGHBOR_COUNT; i++) {
            double neighbor_value = objective_function(&neighbors[i * opt->dim]);
            if (__builtin_expect(neighbor_value < best_neighbor_value, 0)) {
                best_neighbor_value = neighbor_value;
                best_neighbor_idx = i;
            }
        }

        // Stop if no better neighbor found
        if (best_neighbor_idx == -1) {
            break;
        }

        // Move to best neighbor
        for (j = 0; j < opt->dim; j++) {
            current_solution[j] = neighbors[best_neighbor_idx * opt->dim + j];
        }
        current_value = best_neighbor_value;

        // Update global best
        if (best_neighbor_value < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_neighbor_value;
            for (j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = current_solution[j];
            }
        }
    }
}
