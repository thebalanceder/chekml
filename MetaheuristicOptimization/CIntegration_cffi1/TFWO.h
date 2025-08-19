#ifndef TFWO_H
#define TFWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "generaloptimizer.h"

// Optimization parameters
#define N_WHIRLPOOLS_DEFAULT 3
#define N_OBJECTS_PER_WHIRLPOOL_DEFAULT 30
#ifndef PI
#define PI 3.14159265358979323846
#endif
#define ALIGNMENT 32 // 32-byte alignment for AVX2

// Structure for TFWO data (SoA layout for cache efficiency)
typedef struct {
    int n_whirlpools;       // Number of whirlpools
    int n_objects_per_whirlpool; // Objects per whirlpool
    int dim;                // Problem dimension
    int total_positions;    // Total number of positions (whirlpools + objects)
    double *positions;      // Contiguous position array [wp1, wp2, ..., obj1_1, obj1_2, ...]
    double *costs;          // Costs for whirlpools and objects
    double *deltas;         // Deltas for whirlpools and objects
    double *position_sums;  // Cached sum of position components
    double *best_costs;     // Best costs per iteration
    double *mean_costs;     // Mean costs per iteration
    double *temp_d;         // Reusable temporary array d
    double *temp_d2;        // Reusable temporary array d2
    double *temp_RR;        // Reusable temporary array RR
    double *temp_J;         // Reusable temporary array J
} TFWO_Data;

// Utility functions
void seed_rng(uint64_t seed);
double tfwo_rand_double(void);
double fast_cos(double x);
double fast_sin(double x);
double fast_sqrt(double x);
void initialize_whirlpools(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *));
void effects_of_whirlpools(Optimizer *opt, TFWO_Data *data, int iter, double (*objective_function)(double *));
void update_best_whirlpool(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *));
void free_tfwo_data(TFWO_Data *data);

// Main optimization function
void TFWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TFWO_H
