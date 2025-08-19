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
#include "generaloptimizer.h"

// Optimization parameters
#define WO_FEMALE_PROPORTION 0.4
#define WO_HALTON_BASE 7
#define WO_LEVY_BETA 1.5
#define WO_MAX_DIM 100
#define WO_MAX_POP 1000
#define WO_RAND_BUFFER_SIZE 4096

// Xorshift RNG struct
typedef struct {
    uint64_t state;
} Xorshift;

// Context struct for WO-specific state
typedef struct {
    int male_count;
    int female_count;
    int child_count;
    double levy_sigma;
    int temp_indices[WO_MAX_POP];
    double temp_array1[WO_MAX_DIM];
    double temp_array2[WO_MAX_DIM];
    Solution second_best;
    double second_best_pos[WO_MAX_DIM];
    double rand_buffer[WO_RAND_BUFFER_SIZE];
    int rand_index;
    Xorshift rng;
} WOContext;

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WO_H
