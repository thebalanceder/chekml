#ifndef TFWO_H
#define TFWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define N_WHIRLPOOLS_DEFAULT 3
#define N_OBJECTS_PER_WHIRLPOOL_DEFAULT 30
#ifndef PI
#define PI 3.14159265358979323846
#endif

// Structure to represent an object (particle) in a whirlpool
typedef struct {
    double *position; // Position in search space
    double cost;      // Objective function value
    double delta;     // Angular displacement
} TFWO_Object;

// Structure to represent a whirlpool
typedef struct {
    double *position; // Whirlpool center position
    double cost;      // Objective function value
    double delta;     // Angular displacement
    double position_sum; // Cached sum of position components
    int n_objects;    // Number of objects in whirlpool
    TFWO_Object *objects; // Array of objects
} TFWO_Whirlpool;

// Structure for TFWO-specific data
typedef struct {
    int n_whirlpools;       // Number of whirlpools
    int n_objects_per_whirlpool; // Objects per whirlpool
    TFWO_Whirlpool *whirlpools; // Array of whirlpools
    double *best_costs;     // Best costs per iteration
    double *mean_costs;     // Mean costs per iteration
    double *temp_d;         // Reusable temporary array d
    double *temp_d2;        // Reusable temporary array d2
    double *temp_RR;        // Reusable temporary array RR
    double *temp_J;         // Reusable temporary array J
} TFWO_Data;

// Utility functions
double rand_double_tfwo(double min, double max);
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
