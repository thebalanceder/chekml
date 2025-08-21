#ifndef BHA_H
#define BHA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>  // For fixed-width types
#include "generaloptimizer.h"

// Optimization parameters
static const int BHA_POPULATION_SIZE = 50;
static const int BHA_MAX_ITER = 500;

// Fast random number generator state
typedef struct {
    uint64_t state;
} FastRNG;

// Black Hole Algorithm Phases
void initialize_stars(Optimizer *restrict opt, double (*objective_function)(double *));
void update_star_positions(Optimizer *restrict opt, int black_hole_idx);
void replace_with_better_black_hole(Optimizer *restrict opt, int *restrict black_hole_idx);
void new_star_generation(Optimizer *restrict opt, int black_hole_idx, double (*objective_function)(double *));

// Optimization Execution
void BHA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

// Inline utility for fast random double
static inline double fast_rand_double_bha(FastRNG *restrict rng, double min, double max) {
    rng->state ^= rng->state >> 12;
    rng->state ^= rng->state << 25;
    rng->state ^= rng->state >> 27;
    uint64_t r = rng->state * 0x2545F4914F6CDD1DULL;
    double t = (double)(r >> 12) / (1ULL << 52);
    return min + t * (max - min);
}

#ifdef __cplusplus
}
#endif

#endif // BHA_H
