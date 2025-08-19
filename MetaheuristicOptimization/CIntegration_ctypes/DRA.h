#ifndef DRA_H
#define DRA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// 🔧 DRA Optimization Parameters
#define BELIEF_PROFILE_RATE 0.5
#define MIRACLE_RATE 0.5
#define PROSELYTISM_RATE 0.9
#define REWARD_PENALTY_RATE 0.2
#define NUM_GROUPS 5
#ifndef PI
#define PI 3.141592653589793
#endif

// 🌍 DRA Algorithm Phases
void initialize_belief_profiles(Optimizer *opt, ObjectiveFunction objective_function);
void initialize_groups(Optimizer *opt);
void miracle_operator(Optimizer *opt, ObjectiveFunction objective_function);
void proselytism_operator(Optimizer *opt, ObjectiveFunction objective_function);
void reward_penalty_operator(Optimizer *opt, ObjectiveFunction objective_function);
void replacement_operator(Optimizer *opt, ObjectiveFunction objective_function);

// 🚀 Optimization Execution
void DRA_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // DRA_H
