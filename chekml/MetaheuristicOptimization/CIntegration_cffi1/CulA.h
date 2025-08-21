#ifndef CULA_H
#define CULA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include <stdint.h>  // ✅ For uint64_t and UINT64_MAX
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Cultural Algorithm Parameters
#define ALPHA_SCALING 0.1      // Scaling factor for cultural influence
#define ACCEPTANCE_RATIO 0.2   // Ratio of population accepted for culture adjustment

// 🧠 Culture Knowledge Structures
typedef struct {
    double *position;  // Best position found
    double cost;       // Best fitness (cost) found
} SituationalKnowledge;

typedef struct {
    double *min;       // Minimum bounds for each dimension
    double *max;       // Maximum bounds for each dimension
    double *L;         // Lower fitness bound for each dimension
    double *U;         // Upper fitness bound for each dimension
    double *size;      // Size of the normative range (max - min)
} NormativeKnowledge;

typedef struct {
    SituationalKnowledge situational;  // Situational knowledge component
    NormativeKnowledge normative;      // Normative knowledge component
} Culture;

// 🎲 Xorshift RNG State
typedef struct {
    uint64_t state;
} XorshiftState;

// 🎲 Xorshift Utility Functions
inline uint64_t xorshift_next(XorshiftState *state) {
    uint64_t x = state->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state->state = x;
    return x;
}

inline double xorshift_double(XorshiftState *state, double min, double max) {
    uint64_t x = xorshift_next(state);
    double range = max - min;
    return min + range * ((double)x / (double)UINT64_MAX);
}

inline double xorshift_normal(XorshiftState *state, double mean, double stddev) {
    double u, v, s;
    do {
        u = xorshift_double(state, -1.0, 1.0);
        v = xorshift_double(state, -1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    
    s = sqrt(-2.0 * log(s) / s);
    return mean + stddev * (u * s);
}

// 🎲 Comparison function for qsort
int compare_individuals(const void *a, const void *b);

// 🌍 Cultural Algorithm Phases
void initialize_culture(Optimizer *opt, Culture *culture);
void adjust_culture(Optimizer *opt, Culture *culture, int n_accept);
void influence_culture(Optimizer *opt, Culture *culture);

// 🚀 Optimization Execution
void CulA_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // CULA_H
