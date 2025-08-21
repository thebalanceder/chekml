#ifndef HPO_H
#define HPO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h> // SSE2 intrinsics
#ifdef __AVX__
#include <immintrin.h> // AVX intrinsics
#endif
#include "generaloptimizer.h"

// Optimization parameters
#define CONSTRICTION_COEFF 0.1
#define C_PARAM_MAX 0.98
#define TWO_PI 6.283185307179586
#define HPO_ALIGNMENT 16 // SSE2 alignment (bytes), unique to HPO

// Fast random number generator (Xorshift)
typedef struct {
    unsigned long x, y, z, w;
} HPOXorshiftState;

void hpo_xorshift_init(HPOXorshiftState *state, unsigned long seed);
double hpo_xorshift_double(HPOXorshiftState *state);

// Helper for quicksort
void hpo_quicksort_indices(double *arr, int *indices, int low, int high);

// HPO Algorithm Phases
void hpo_update_positions(Optimizer *opt, int iter, double c_factor, HPOXorshiftState *rng, double *xi, double *dist, int *idxsortdist, double *r1, double *r3, char *idx, double *z);

// Optimization Execution
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HPO_H
