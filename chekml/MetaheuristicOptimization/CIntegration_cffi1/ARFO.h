#ifndef ARFO_H
#define ARFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization Parameters
#define BRANCHING_THRESHOLD 0.6
#define MAX_BRANCHING 5
#define MIN_BRANCHING 1
#define INITIAL_STD 1.0
#define FINAL_STD 0.01
#define MAX_ELONGATION 0.1
#define LOCAL_INERTIA 0.5
#define ELIMINATION_PERCENTILE 10.0
#define AUXIN_NORMALIZATION_FACTOR 1.0
#define MEDIAN_THRESHOLD 0.5

// Struct for fitness sorting
typedef struct {
    double fitness;
    int index;
} FitnessIndex;

// Fast random number generator state for ARFO
typedef struct {
    unsigned long long state;
} ARFO_FastRNG;

// ARFO Algorithm Phases
void regrowth_phase(Optimizer *restrict opt, int *restrict topology, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted, ARFO_FastRNG *restrict rng);
void branching_phase(Optimizer *restrict opt, int t, double *restrict auxin, double *restrict fitness, double *restrict new_roots, int *restrict new_root_count, FitnessIndex *restrict fitness_indices, ARFO_FastRNG *restrict rng);
void lateral_growth_phase(Optimizer *restrict opt, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted, double *restrict new_roots, ARFO_FastRNG *restrict rng);
void elimination_phase_arfo(Optimizer *restrict opt, double *restrict auxin, double *restrict fitness, double *restrict auxin_sorted);
void replenish_phase(Optimizer *restrict opt, double *restrict fitness, double (*objective_function)(double *), ARFO_FastRNG *restrict rng);
void ARFO_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ARFO_H
