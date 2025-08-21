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
#define BHC_DELTA 0.1
#define CONVERGENCE_THRESHOLD 1e-6

// Group structure
typedef struct {
    int *members;           // Array of indices of spider monkeys
    int size;               // Number of members
    double *leader_position; // Local leader position
    double leader_fitness;  // Local leader fitness
    int leader_count;       // Stagnation counter
} Group;

// SMO-specific optimizer struct
typedef struct {
    Optimizer *base_optimizer;    // Pointer to the base Optimizer
    ObjectiveFunction objective_function; // Objective function pointer
    Group *groups;               // Array of groups
    int num_groups;              // Number of groups
    double *global_leader;       // Global leader position
    double global_leader_fitness; // Global leader fitness
    int global_leader_count;     // Stagnation counter
    double *temp_position;       // Reusable temporary position array
} SMOOptimizer;

// SMO Algorithm Phases
void local_leader_phase(SMOOptimizer *smo);
void global_leader_phase(SMOOptimizer *smo);
void local_leader_decision(SMOOptimizer *smo);
void global_leader_decision(SMOOptimizer *smo);
void beta_hill_climbing(SMOOptimizer *smo, int idx, double *new_position, double *new_fitness);

// Optimization Execution
void SMO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SMO_H
