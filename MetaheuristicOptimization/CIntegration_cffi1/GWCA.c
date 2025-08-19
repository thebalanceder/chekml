#include "GWCA.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics

#define ALIGNMENT 32

// Fast PRNG using xorshift32
static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline double fast_rand(uint32_t* state) {
    return (double)xorshift32(state) / (double)UINT32_MAX;
}

// Aligned memory allocation
void* aligned_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0) return NULL;
    return ptr;
}

int compare_fitness_gwca(const void *a, const void *b) {
    const Solution *sol_a = (const Solution*)a;
    const Solution *sol_b = (const Solution*)b;
    return (sol_a->fitness > sol_b->fitness) - (sol_a->fitness < sol_b->fitness);
}

void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    double best_overall = INFINITY;
    Solution Worker1, Worker2, Worker3;

    // Sort and initialize top workers
    qsort(opt->population, opt->population_size, sizeof(Solution), compare_fitness_gwca);
    Worker1 = opt->population[0];
    Worker2 = opt->population[1];
    Worker3 = opt->population[2];
    best_overall = Worker1.fitness;

    const int LNP = (int)ceil(opt->population_size * E);
    const double C_slope = (CMAX - CMIN) / opt->max_iter;
    const double inv_max_rand = 1.0 / (double)UINT32_MAX;

    // Thread-local PRNG states (simulate threads in serial for now)
    uint32_t* seeds = (uint32_t*)aligned_malloc(opt->population_size * sizeof(uint32_t));
    for (int i = 0; i < opt->population_size; i++)
        seeds[i] = 2463534242u ^ (i * 747796405u);

    // Memory buffers for SIMD
    __m256d* influence_buffer = (__m256d*)aligned_malloc(opt->dim * sizeof(__m256d));

    // Main loop
    for (int t = 1; t <= opt->max_iter; t++) {
        const double C = CMAX - C_slope * t;

        for (int i = 0; i < opt->population_size; i++) {
            double* position = opt->population[i].position;
            uint32_t* seed = &seeds[i];
            double r1 = fast_rand(seed);
            double r2 = fast_rand(seed);

            if (i < LNP) {
                double F = (M * G * r1) / (P * Q * (1 + t));
                for (int d = 0; d < opt->dim; d++) {
                    double delta = F * (xorshift32(seed) & 1 ? 1.0 : -1.0) * C;
                    position[d] += delta;
                }
            } else {
                // SIMD vectorized update for position
                for (int d = 0; d < opt->dim; d += 4) {
                    // Prefetch the next cache line
                    _mm_prefetch((char*)&opt->population[i].position[d+4], _MM_HINT_T0);
                    __m256d influence = _mm256_set1_pd((Worker1.position[d] + Worker2.position[d] + Worker3.position[d]) / 3.0);
                    __m256d delta = _mm256_set1_pd(r2 * C);
                    __m256d updated_position = _mm256_add_pd(influence, delta);
                    _mm256_storeu_pd(&position[d], updated_position);
                }
            }

            enforce_bound_constraints(opt);
            opt->population[i].fitness = objective_function(position);

            if (opt->population[i].fitness < Worker1.fitness) {
                Worker3 = Worker2;
                Worker2 = Worker1;
                Worker1 = opt->population[i];
            }
        }

        if (Worker1.fitness < best_overall) {
            best_overall = Worker1.fitness;
            memcpy(opt->best_solution.position, Worker1.position, opt->dim * sizeof(double));
            opt->best_solution.fitness = Worker1.fitness;
        }

        printf("Iteration %d: Best Fitness = %f\n", t, Worker1.fitness);
    }

    free(seeds);
    free(influence_buffer);
}
