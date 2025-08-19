#ifndef CULA_H
#define CULA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Cultural Algorithm Parameters
#define ALPHA_SCALING 0.1f      // Scaling factor for cultural influence
#define ACCEPTANCE_RATIO 0.2f   // Ratio of population accepted for culture adjustment
#define CULA_TOL 1e-6f          // Convergence tolerance

// Culture Knowledge Structures
typedef struct {
    cl_mem position;  // GPU buffer for best position (dim * sizeof(float))
    float cost;       // Best fitness (cost) found
} SituationalKnowledge;

typedef struct {
    cl_mem min;       // GPU buffer for minimum bounds (dim * sizeof(float))
    cl_mem max;       // GPU buffer for maximum bounds (dim * sizeof(float))
    cl_mem L;         // GPU buffer for lower fitness bounds (dim * sizeof(float))
    cl_mem U;         // GPU buffer for upper fitness bounds (dim * sizeof(float))
    cl_mem size;      // GPU buffer for normative range sizes (dim * sizeof(float))
} NormativeKnowledge;

typedef struct {
    SituationalKnowledge situational;
    NormativeKnowledge normative;
} Culture;

// Optimization Execution
void CulA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // CULA_H
