#ifndef CULA_H
#define CULA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
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

// 🎲 Utility Functions
double rand_double_cula(double min, double max);
double rand_normal_cula(double mean, double stddev);
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
