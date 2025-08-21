#ifndef CUCS_H
#define CUCS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CS_POPULATION_SIZE 32 // Aligned for SIMD
#define CS_MAX_ITER 100
#define CS_PA 0.25
#define CS_BETA 1.5
#define CS_STEP_SCALE 0.01

// Precomputed constants
#define CS_PI 3.141592653589793
#define CS_SIGMA 0.696066
#define INV_RAND_MAX (1.0 / 4294967295.0)

// Fast LCG random number generator state
typedef struct {
    uint64_t state;
} LCG_Rand;

// Function declarations
void initialize_nests(Optimizer *restrict opt);
void evaluate_nests(Optimizer *restrict opt, double (*objective_function)(double *));
void get_cuckoos(Optimizer *restrict opt);
void empty_nests(Optimizer *restrict opt);
void CucS_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

// Inline functions
static inline uint32_t lcg_next(LCG_Rand *restrict rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng->state >> 32);
}

static inline double cucs_rand_double(LCG_Rand *restrict rng, double min, double max) {
    return min + (max - min) * (lcg_next(rng) * INV_RAND_MAX);
}

static inline void fast_enforce_bounds(Optimizer *restrict opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            pos[j] = fmax(min, fmin(max, pos[j]));
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif
