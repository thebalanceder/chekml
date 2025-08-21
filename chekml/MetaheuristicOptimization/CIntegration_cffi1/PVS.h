#ifndef PVS_H
#define PVS_H

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
#define DISTRIBUTION_INDEX 20
#define X_GAMMA 0.1
#define LOOKUP_SIZE 1024
#define CACHE_LINE 64

// 🌌 PVS-specific data (cache-aligned)
typedef struct __attribute__((aligned(CACHE_LINE))) {
    double *center; // Vortex center
    double *obj_vals; // Fitness values
    double *prob; // Roulette wheel probabilities
    double *sol; // Temporary solution for crossover
    double *mutated; // Temporary solution for mutation
    double prob_mut; // Mutation probability
    double prob_cross; // Crossover probability
    double *bound_diffs; // Precomputed (upper - lower) bounds
    uint64_t rng_state; // LCG state
    double normal_lookup[LOOKUP_SIZE]; // Precomputed normal variates
    double gamma_lookup[LOOKUP_SIZE]; // Precomputed gamma inverse
} PVSData;

// 🚀 Fast random number generation
inline uint64_t fast_rng(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

inline double fast_rand_double_pvs(uint64_t *state, double min, double max) {
    return min + (max - min) * ((fast_rng(state) >> 11) * (1.0 / (1ULL << 53)));
}

// 🚀 PVS Algorithm Phases
void initialize_vortex(Optimizer *opt, PVSData *data);
void first_phase(Optimizer *opt, PVSData *data, int iteration, double radius);
void second_phase(Optimizer *opt, PVSData *data, int iteration, ObjectiveFunction objective_function);
void polynomial_mutation(Optimizer *opt, PVSData *data, double *__restrict solution, double *__restrict mutated, int *__restrict state);

// 🌌 Optimization Execution
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // PVS_H
