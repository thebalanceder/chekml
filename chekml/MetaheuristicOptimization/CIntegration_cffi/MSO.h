#ifndef MSO_H
#define MSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define MSO_P_EXPLORE 0.2
#define MSO_MAX_P_EXPLORE 0.8
#define MSO_MIN_P_EXPLORE 0.1
#define MSO_PERTURBATION_SCALE 10.0
#define MSO_INV_RAND_MAX (1.0 / 4294967296.0) // Adjusted for Xorshift (2^32)

// 🐒 MSO-specific Xorshift RNG state
typedef struct {
    unsigned long x, y, z, w;
} MSO_XorshiftState;

// 🐒 MSA Algorithm Functions
void mso_xorshift_init(MSO_XorshiftState *state, unsigned long seed);

// Fast Xorshift random double
inline double mso_xorshift_double(MSO_XorshiftState *state, double min, double max) {
    unsigned long t = state->x ^ (state->x << 11);
    state->x = state->y; state->y = state->z; state->z = state->w;
    state->w ^= (state->w >> 19) ^ (t ^ (t >> 8));
    return min + (max - min) * (state->w * MSO_INV_RAND_MAX);
}

// Fast normal distribution (Box-Muller transform)
inline double mso_xorshift_normal(MSO_XorshiftState *state, double mean, double stddev) {
    double u1 = mso_xorshift_double(state, 0.0, 1.0);
    double u2 = mso_xorshift_double(state, 0.0, 1.0);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z;
}

void mso_update_positions(Optimizer *opt, int iter, MSO_XorshiftState *rng_state);
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // MSO_H
