#ifndef POA_H
#define POA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define STEP_SIZE 0.1

// 🌱 Utility function for random number generation
static inline __attribute__((always_inline)) double rand_double_poa(double min, double max) {
    // Fast LCG for random numbers
    static unsigned long seed = 1;
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = (double)(seed >> 32) / (double)(1ULL << 32);
    return min + (max - min) * r;
}

// 🌱 POA Algorithm Phases
static inline __attribute__((always_inline)) void initialize_population_poa(Optimizer *restrict opt);
void POA_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // POA_H
