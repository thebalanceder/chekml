#ifndef RDA_H
#define RDA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>
#include "generaloptimizer.h"

// Optimization parameters
#define STEP_SIZE 0.1
#define P_EXPLORATION 0.1
#define INV_STEP_SIZE (1.0 / STEP_SIZE)
#define STEP_SIZE_2 (2.0 * STEP_SIZE)

// Fast Xorshift random number generator state
typedef struct {
    uint32_t a;
} XorshiftState_RDA;

// Additional data for RDA (cannot modify Optimizer struct)
typedef struct {
    XorshiftState_RDA rng_state;  // RNG state for fast random number generation
    double* fitness;          // Temporary array for fitness values [population_size]
} RDAData;

// Macros for bounds enforcement (branchless)
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// Inline Xorshift random number generator (faster than rand())
#define XORSHIFT_NEXT(state) ({\
    uint32_t x = (state)->a;\
    x ^= x << 13;\
    x ^= x >> 17;\
    x ^= x << 5;\
    (state)->a = x;\
    x;})

// Generate a random double in [0, 1)
#define RAND_DOUBLE(state) ((double)XORSHIFT_NEXT(state) / (double)UINT32_MAX)

// Generate a random double in [min, max)
#define RAND_RANGE(state, min, max) ((min) + ((max) - (min)) * RAND_DOUBLE(state))

// Main optimization function (matches the optimize function pointer signature)
void RDA_optimize(void* optimizer, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // RDA_H
