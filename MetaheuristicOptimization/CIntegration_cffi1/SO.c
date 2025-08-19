#include "SO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

// Xorshift RNG state for SIMD
static __m128i xorshift_state[4]; // 4 lanes for SSE

// Forward declaration for xorshift_rand
static inline unsigned long xorshift_rand(void);

// Initialize Xorshift seed
static inline void init_xorshift(unsigned long seed) {
    seed = seed ? seed : 1;
    for (int i = 0; i < 4; i++) {
        xorshift_state[i] = _mm_set1_epi32((unsigned int)(seed + i));
    }
}

// SIMD Xorshift random number generator (4 doubles at a time)
static inline __m128d rand_double_so_simd(double min, double max) {
    __m128i x = xorshift_state[0];
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
    x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
    xorshift_state[0] = x;
    __m128d t = _mm_cvtepi32_pd(_mm_and_si128(x, _mm_set1_epi32(0x7FFFFFFF)));
    t = _mm_div_pd(t, _mm_set1_pd((double)0x7FFFFFFF));
    __m128d range = _mm_set1_pd(max - min);
    __m128d min_vec = _mm_set1_pd(min);
    return _mm_add_pd(min_vec, _mm_mul_pd(t, range));
}

// Precomputed trigonometric table
static double trig_table_cos[TRIG_TABLE_SIZE];
static double trig_table_sin[TRIG_TABLE_SIZE];
static void init_trig_table() {
    for (int i = 0; i < TRIG_TABLE_SIZE; i++) {
        double theta = 2.0 * M_PI * (double)i / TRIG_TABLE_SIZE;
        trig_table_cos[i] = cos(theta);
        trig_table_sin[i] = sin(theta);
    }
}

// 🌊 Initialize Population (SIMD for dim=2)
void initialize_population(Optimizer *opt) {
    __m128d min_vec = _mm_set_pd(opt->bounds[2], opt->bounds[0]);
    __m128d range_vec = _mm_set_pd(opt->bounds[3] - opt->bounds[2], opt->bounds[1] - opt->bounds[0]);
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        if (opt->dim == 2) {
            __m128d rand = rand_double_so_simd(0.0, 1.0);
            __m128d result = _mm_add_pd(min_vec, _mm_mul_pd(rand, range_vec));
            _mm_storeu_pd(pos, result);
        } else {
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = opt->bounds[2 * j] + (opt->bounds[2 * j + 1] - opt->bounds[2 * j]) * 
                         ((double)(xorshift_rand() & 0x7FFFFFFF) / (double)0x7FFFFFFF);
            }
        }
        opt->population[i].fitness = DBL_MAX;
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "initialize_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// Fast Xorshift random number generator
static inline unsigned long xorshift_rand() {
    unsigned long x = _mm_cvtsi128_si32(xorshift_state[0]);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state[0] = _mm_set1_epi32((unsigned int)x);
    return x;
}

// 🌊 Spiral Movement Phase (SIMD for dim=2, precomputed trig)
void spiral_movement_phase(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double r = (double)(xorshift_rand() & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        int trig_idx = (xorshift_rand() % TRIG_TABLE_SIZE);
        double ct = trig_table_cos[trig_idx], st = trig_table_sin[trig_idx];
        if (opt->dim == 2) {
            __m128d pos_vec = _mm_loadu_pd(pos);
            __m128d dir_vec = _mm_set_pd(st, ct);
            __m128d step = _mm_mul_pd(_mm_set1_pd(SPIRAL_STEP * r), dir_vec);
            _mm_storeu_pd(pos, _mm_add_pd(pos_vec, step));
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double dir = j < 2 ? (j == 0 ? ct : st) : 
                            ((double)(xorshift_rand() & 0x7FFFFFFF) / (double)0x7FFFFFFF * 2.0 - 1.0);
                pos[j] += SPIRAL_STEP * r * dir;
            }
        }
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "spiral_movement_phase: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🌊 Update and Sort Population
void update_and_sort_population(Optimizer *opt, double (*objective_function)(double *)) {
    // Parallel fitness evaluation with static scheduling
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Update best solution
    double best_fitness = opt->best_solution.fitness;
    int best_idx = -1;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[best_idx].position, sizeof(double) * opt->dim);
    }

    // Sort population by fitness using qsort on indices
    typedef struct {
        int index;
        double fitness;
    } IndexedSolution;

    #define MAX_POP_SIZE 1000
    #define MAX_DIM 100
    IndexedSolution indexed[MAX_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        indexed[i].index = i;
        indexed[i].fitness = opt->population[i].fitness;
    }

    int compare_by_fitness(const void *a, const void *b) {
        const IndexedSolution *ia = (const IndexedSolution *)a;
        const IndexedSolution *ib = (const IndexedSolution *)b;
        return (ia->fitness > ib->fitness) - (ia->fitness < ib->fitness);
    }
    qsort(indexed, opt->population_size, sizeof(IndexedSolution), compare_by_fitness);

    // Reorder population using index mapping
    double temp_pos[MAX_POP_SIZE * MAX_DIM];
    double temp_fits[MAX_POP_SIZE];
    for (int i = 0; i < opt->population_size; i++) {
        temp_fits[i] = opt->population[i].fitness;
        memcpy(temp_pos + i * opt->dim, opt->population[i].position, sizeof(double) * opt->dim);
    }
    for (int i = 0; i < opt->population_size; i++) {
        int old_idx = indexed[i].index;
        opt->population[i].fitness = temp_fits[old_idx];
        memcpy(opt->population[i].position, temp_pos + old_idx * opt->dim, sizeof(double) * opt->dim);
    }
    fprintf(stderr, "update_and_sort_population: population[0].position=%p\n", (void*)opt->population[0].position);
}

// 🚀 Main Optimization Function
void SO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in SO_optimize\n");
        return;
    }
    if (opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid population_size, dim, or max_iter\n");
        return;
    }
    if (opt->population_size > MAX_POP_SIZE || opt->dim > MAX_DIM) {
        fprintf(stderr, "Error: Population size (%d) or dimension (%d) exceeds maximum\n", 
                opt->population_size, opt->dim);
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null position in population[%d]\n", i);
            return;
        }
    }
    init_xorshift((unsigned long)time(NULL));
    init_trig_table();
    fprintf(stderr, "SO_optimize: starting, population[0].position=%p\n", (void*)opt->population[0].position);
    initialize_population(opt);
    for (int iter = 0; iter < opt->max_iter; iter++) {
        spiral_movement_phase(opt);
        update_and_sort_population(opt, objective_function);
        enforce_bound_constraints(opt);
    }
    fprintf(stderr, "SO_optimize: completed, population[0].position=%p, best_solution.position=%p\n", 
            (void*)opt->population[0].position, (void*)opt->best_solution.position);
}
