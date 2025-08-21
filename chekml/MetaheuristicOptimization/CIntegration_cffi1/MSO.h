#ifndef MSO_H
#define MSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h> // AVX/SSE intrinsics
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define MSO_P_EXPLORE 0.2
#define MSO_MAX_P_EXPLORE 0.8
#define MSO_MIN_P_EXPLORE 0.1
#define MSO_PERTURBATION_SCALE 10.0
#define MSO_RNG_BUFFER_SIZE 1024 // Preallocated random numbers

// 🐒 MSO-specific Xorshift128+ RNG state
typedef struct {
    unsigned long long s[2];
} MSO_XorshiftState;

// 🐒 Preallocated random number buffer
typedef struct {
    double *uniform; // Uniform random numbers [0,1]
    double *normal;  // Normal random numbers (mean=0, stddev=1)
    size_t uniform_idx;
    size_t normal_idx;
    size_t size;
} MSO_RNGBuffer;

// 🐒 MSA Algorithm Functions
void mso_xorshift_init(MSO_XorshiftState *state, unsigned long long seed);
static inline unsigned long long mso_xorshift_next(MSO_XorshiftState *state);
static inline double mso_xorshift_double(MSO_RNGBuffer *buffer, MSO_XorshiftState *state);
static inline double mso_xorshift_normal(MSO_RNGBuffer *buffer, MSO_XorshiftState *state);
void mso_rng_buffer_init(MSO_RNGBuffer *buffer, MSO_XorshiftState *state, size_t size);
void mso_rng_buffer_free(MSO_RNGBuffer *buffer);
void mso_update_positions(Optimizer *opt, int iter, MSO_XorshiftState *rng_state, MSO_RNGBuffer *rng_buffer);
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // MSO_H
