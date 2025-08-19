#ifndef BDFO_H
#define BDFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>  // ✅ For uint32_t
#include <immintrin.h>  // ✅ For SIMD intrinsics (SSE/AVX)
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define BDFO_EXPLORATION_FACTOR 0.5f
#define BDFO_ADJUSTMENT_RATE 0.3f
#define BDFO_ELIMINATION_RATIO 0.23f
#define BDFO_PERTURBATION_SCALE 0.1f
#define BDFO_EXPLORATION_PROB 0.3f
#define BDFO_SEGMENT_FACTOR 0.5f
#define POP_SIZE_MAX 1000  // ✅ Maximum population size for stack allocation
#define DIM_MAX 100       // ✅ Maximum dimensions for stack allocation

// ⚙️ Fast Xorshift RNG state
typedef struct {
    uint32_t state;
} Xorshift32;

// 🐬 Inline RNG functions
static inline uint32_t xorshift32_next(Xorshift32 *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static inline float bdfo_rand_float(Xorshift32 *rng, float min, float max) {
    return min + (max - min) * ((float)xorshift32_next(rng) / (float)0xFFFFFFFF);
}

// 🐬 BDFO Algorithm Phases
void bdfo_bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *));
void bdfo_exploration_phase(Optimizer *opt);
void bdfo_elimination_phase(Optimizer *opt);

// 🚀 Optimization Execution
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BDFO_H
