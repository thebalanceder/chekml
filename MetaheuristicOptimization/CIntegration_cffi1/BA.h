#ifndef BA_H
#define BA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h> // AVX intrinsics
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h> // For uint64_t
#include <omp.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define LOUDNESS 1.0        // Initial loudness (A)
#define PULSE_RATE 1.0      // Initial pulse rate (r0)
#define ALPHA_BA 0.97          // Loudness decay factor
#define GAMMA 0.1           // Pulse rate increase factor
#define FREQ_MIN 0.0        // Minimum frequency
#define FREQ_MAX 2.0        // Maximum frequency
#define LOCAL_SEARCH_SCALE 0.1  // Scale for local search step
#define ALIGNMENT 32        // AVX alignment (bytes)

// 🦇 Xorshift128+ RNG state
typedef struct {
    uint64_t s[2];
} XorshiftState_Ba;

// 🦇 Inline utility functions
static inline double xorshift_double_ba(XorshiftState_Ba *restrict state) {
    uint64_t x = state->s[0];
    uint64_t y = state->s[1];
    state->s[0] = y;
    x ^= x << 23;
    state->s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return (state->s[1] + y) * (1.0 / (1ULL << 53));
}

// 🦇 RNG initialization
void init_xorshift_state(XorshiftState_Ba *restrict state, uint64_t seed);

// 🦇 Bat Algorithm Phases
void bat_frequency_update(Optimizer *restrict opt, double *restrict freq, double *restrict velocities, XorshiftState_Ba *restrict rng_states);
void bat_local_search(Optimizer *restrict opt, double *restrict freq, double pulse_rate, double loudness, XorshiftState_Ba *restrict rng_states);
void bat_update_solutions(Optimizer *restrict opt, double *restrict freq, double loudness, double (*objective_function)(double *), XorshiftState_Ba *restrict rng_states);

// 🚀 Optimization Execution
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BA_H
