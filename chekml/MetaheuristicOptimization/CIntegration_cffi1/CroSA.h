#ifndef CROSA_H
#define CROSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define AWARENESS_PROBABILITY 0.1f
#define FLIGHT_LENGTH 2.0f

// 🌍 Fast Xorshift RNG state
typedef struct {
    uint64_t state;
} XorshiftState_crosa;

// 🐦 CSA Algorithm Phases (inlined where possible)
void initialize_population_crosa(Optimizer *opt, Solution *memory, double (*objective_function)(double *), XorshiftState_crosa *rng);
void update_positions_and_memory(Optimizer *opt, Solution *memory, double (*objective_function)(double *), XorshiftState_crosa *rng);

// 🚀 Optimization Execution
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *));

// 🌍 Inline utility functions
static inline uint64_t xorshift64(XorshiftState_crosa *state) {
    uint64_t x = state->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double fast_rand_double(XorshiftState_crosa *state, double min, double max) {
    return min + (max - min) * ((double)xorshift64(state) / UINT64_MAX);
}

#ifdef __cplusplus
}
#endif

#endif // CROSA_H
