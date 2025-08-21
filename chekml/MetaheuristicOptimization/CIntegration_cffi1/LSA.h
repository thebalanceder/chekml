#ifndef LSA_H
#define LSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MAX_CHANNEL_TIME 5
#define LSA_ENERGY_FACTOR 2.05
#define FOCKING_PROB 0.002
#define DIRECTION_STEP 0.005
#define RAND_NORMAL_MAX 10.0
#define RAND_EXP_MAX 20.0
#define MIN_LAMBDA 1e-5
#define STALL_LIMIT 10

// Fast LCG random number generator
extern uint64_t lcg_state;
inline double rand_double_lsa(void) {
    lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;
    return (double)(lcg_state >> 32) / 4294967296.0;
}

// Core LSA function
void LSA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // LSA_H
