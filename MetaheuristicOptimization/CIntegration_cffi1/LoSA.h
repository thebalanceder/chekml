#ifndef LOSA_H
#define LOSA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // For malloc/free
#include <stdint.h>  // For uint64_t
#include <immintrin.h>  // For AVX2 intrinsics
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define LOSA_POPULATION_SIZE 50
#define LOSA_MAX_ITERATIONS 100
#define LOSA_STEP_SIZE 0.1
#define LOSA_SIMD_WIDTH 4  // AVX2: 4 doubles per vector (256 bits)

// Xorshift128+ state for fast random number generation
typedef struct {
    uint64_t s[2];
} LoSA_XorshiftState;

// Initialize Xorshift state
static inline void LoSA_xorshift_init(LoSA_XorshiftState *state, uint64_t seed) {
    state->s[0] = seed;
    state->s[1] = seed ^ 0xdeadbeef;
}

// Fast Xorshift128+ random number generator (returns uint64_t)
static inline uint64_t LoSA_xorshift_next(LoSA_XorshiftState *state) {
    uint64_t s0 = state->s[0];
    uint64_t s1 = state->s[1];
    state->s[0] = s1;
    s0 ^= s0 << 23;
    state->s[1] = s0 ^ s1 ^ (s0 >> 17) ^ (s1 >> 26);
    return state->s[1] + s1;
}

// Generate random double in [0, 1)
static inline double LoSA_xorshift_double(LoSA_XorshiftState *state) {
    return (double)(LoSA_xorshift_next(state) >> 11) / (double)(1ULL << 53);
}

// Aligned memory allocation
static inline void *aligned_malloc(size_t size, size_t alignment) {
    return _mm_malloc(size, alignment);
}

// Aligned memory free
static inline void aligned_free(void *ptr) {
    _mm_free(ptr);
}

// LSA Algorithm Functions
void LoSA_initialize_population(Optimizer *opt, LoSA_XorshiftState *rng);
void LoSA_update_positions(Optimizer *opt, LoSA_XorshiftState *rng, double *rand_buffer);

// Optimization Execution
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // LOSA_H
