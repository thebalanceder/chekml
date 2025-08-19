#ifndef BSO_H
#define BSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdint.h>  // For uint64_t
#include <time.h>    // For time
#include <immintrin.h>  // For SIMD intrinsics
#include "generaloptimizer.h"

// Optimization parameters
#define BSO_PERTURBATION_SCALE 0.1

// SIMD configuration
#if defined(__AVX__)
#define BSO_SIMD_WIDTH 4
#define BSO_VEC_TYPE __m256d
#define BSO_VEC_SET1 _mm256_set1_pd
#define BSO_VEC_ADD _mm256_add_pd
#define BSO_VEC_MUL _mm256_mul_pd
#define BSO_VEC_MIN _mm256_min_pd
#define BSO_VEC_MAX _mm256_max_pd
#define BSO_VEC_LOAD _mm256_loadu_pd
#define BSO_VEC_STORE _mm256_storeu_pd
#elif defined(__SSE2__)
#define BSO_SIMD_WIDTH 2
#define BSO_VEC_TYPE __m128d
#define BSO_VEC_SET1 _mm_set1_pd
#define BSO_VEC_ADD _mm_add_pd
#define BSO_VEC_MUL _mm_mul_pd
#define BSO_VEC_MIN _mm_min_pd
#define BSO_VEC_MAX _mm_max_pd
#define BSO_VEC_LOAD _mm_loadu_pd
#define BSO_VEC_STORE _mm_storeu_pd
#else
#define BSO_SIMD_WIDTH 1  // Scalar fallback
#endif

// Macro for bounds clipping
#define BSO_CLIP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// BSO-specific Xorshift PRNG state
typedef struct {
    uint64_t state;
} BSOXorshiftState;

// BSO Algorithm Phases
void bso_initialize_population(Optimizer *opt, BSOXorshiftState *rng);
void bso_local_search(Optimizer *opt, BSOXorshiftState *rng);

// Optimization Execution
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BSO_H
