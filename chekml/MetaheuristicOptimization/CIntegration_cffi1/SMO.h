#ifndef SMO_H
#define SMO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define PERTURBATION_RATE 0.1
#define LOCAL_LEADER_LIMIT 50
#define GLOBAL_LEADER_LIMIT 1500
#define MAX_GROUPS 5
#define SMO_MAX_POPULATION 1000
#define SMO_MAX_DIM 100
#define BHC_DELTA 0.1
#define SMO_CONVERGENCE_THRESHOLD 1e-6

// Group structure (flattened for cache efficiency)
typedef struct {
    int members[SMO_MAX_POPULATION]; // Array of indices of spider monkeys
    int size;                        // Number of members
    double leader_position[SMO_MAX_DIM]; // Local leader position
    double leader_fitness;           // Local leader fitness
    int leader_count;                // Stagnation counter
} Group;

// SMO-specific optimizer struct
typedef struct {
    Optimizer *base_optimizer;    // Pointer to the base Optimizer
    ObjectiveFunction objective_function; // Objective function pointer
    Group groups[MAX_GROUPS];    // Fixed-size array of groups
    int num_groups;              // Number of groups
    double global_leader[SMO_MAX_DIM]; // Global leader position
    double global_leader_fitness; // Global leader fitness
    int global_leader_count;     // Stagnation counter
    double temp_position[SMO_MAX_DIM]; // Reusable temporary position array
    unsigned int rng_state;      // LCG state for fast RNG
} SMOOptimizer;

// Optimization Execution
void SMO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SMO_H
