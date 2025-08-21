#ifndef WO_H
#define WO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "generaloptimizer.h"

// Optimization parameters
#define WO_FEMALE_PROPORTION 0.4
#define WO_HALTON_BASE 7
#define WO_LEVY_BETA 1.5

// Xorshift RNG struct
typedef struct {
    uint64_t state;
} Xorshift;

// Xorshift initialization
static inline void xorshift_init(Xorshift *rng, uint64_t seed) {
    rng->state = seed ? seed : 88172645463325252ULL;
}

// Context struct for WO-specific state
typedef struct {
    int male_count;
    int female_count;
    int child_count;
    double levy_sigma;
    int *temp_indices;
    double *temp_array1;
    double *temp_array2;
    Solution second_best;
} WOContext;

// Helper functions
static inline double WO_rand_double(double min, double max) {
    static Xorshift rng = {0};
    static int initialized = 0;
    if (!initialized) {
        xorshift_init(&rng, (uint64_t)time(NULL));
        initialized = 1;
    }
    rng.state ^= rng.state >> 12;
    rng.state ^= rng.state << 25;
    rng.state ^= rng.state >> 27;
    uint64_t tmp = rng.state * 2685821657736338717ULL;
    return min + (max - min) * ((double)(tmp >> 32) / UINT32_MAX);
}

static inline double WO_halton_sequence(int index, int base) {
    double result = 0.0;
    double f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i = i / base;
        f /= base;
    }
    return result;
}

void WO_levy_flight(double *step, int dim, double sigma);

// WO Algorithm Phases
void WO_migration_phase(Optimizer *opt, WOContext *ctx, double beta, double r3);
void WO_male_position_update(Optimizer *opt, WOContext *ctx);
void WO_female_position_update(Optimizer *opt, WOContext *ctx, double alpha);
void WO_child_position_update(Optimizer *opt, WOContext *ctx);
void WO_position_adjustment_phase(Optimizer *opt, WOContext *ctx, double R);
void WO_exploitation_phase(Optimizer *opt, WOContext *ctx, double beta);

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WO_H
