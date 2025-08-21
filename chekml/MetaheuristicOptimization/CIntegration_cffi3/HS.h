#ifndef HS_H
#define HS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Harmony Search parameters
#define HS_MEMORY_SIZE 256
#define HS_LOCAL_WORK_SIZE 64
#define HS_HARMONY_MEMORY_CONSIDERING_RATE 0.95f
#define HS_PITCH_ADJUSTMENT_RATE 0.3f
#define HS_BANDWIDTH 0.2f

// Function declarations
void HS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // HS_H
