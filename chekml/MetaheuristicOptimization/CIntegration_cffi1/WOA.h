#ifndef WOA_H
#define WOA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h> // AVX2 intrinsics
#include <math.h>
#include <stdlib.h>
#include <stdint.h> // For uint64_t
#include "generaloptimizer.h"

// Optimization parameters
#define WOA_A_INITIAL 2.0
#define WOA_A2_INITIAL -1.0
#define WOA_A2_FINAL -2.0
#define WOA_B 1.0
#define WOA_PI 3.14159265358979323846

// Fast random number generator (xorshift128+)
typedef struct {
    uint64_t state[2];
} woa_rng_state_t;

void woa_init_rng(woa_rng_state_t *state, uint64_t seed);
double woa_rand_double(woa_rng_state_t *state);

// Fast math approximations
double woa_fast_exp(double x);
double woa_fast_cos(double x);

// WOA Algorithm Phases
void initialize_positions(Optimizer *opt, woa_rng_state_t *rng_states);
void update_leader(Optimizer *opt, double (*objective_function)(double *));
void update_positions(Optimizer *opt, int t, woa_rng_state_t *rng_states);

// Optimization Execution
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WOA_H
