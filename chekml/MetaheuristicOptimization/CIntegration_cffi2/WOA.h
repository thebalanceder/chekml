#ifndef WOA_H
#define WOA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define WOA_A_INITIAL 2.0
#define WOA_A2_INITIAL -1.0
#define WOA_A2_FINAL -2.0
#define WOA_B 1.0

// Fast random number generator (xorshift)
unsigned int xorshift32_woa(unsigned int *state);
double fast_rand_double(double min, double max, unsigned int *rng_state);

// WOA Algorithm Phases
void initialize_positions(Optimizer *opt, unsigned int *rng_state);
void update_leader(Optimizer *opt, double (*objective_function)(double *));
void update_positions_woa(Optimizer *opt, int t, unsigned int *rng_state);

// Optimization Execution
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WOA_H
