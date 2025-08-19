#ifndef HS_H
#define HS_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Harmony Search parameters
#define HS_MEMORY_SIZE 100
#define HS_MAX_ITERATIONS 100
#define HS_HARMONY_MEMORY_CONSIDERING_RATE 0.95
#define HS_PITCH_ADJUSTMENT_RATE 0.3
#define HS_BANDWIDTH 0.2

// Function declarations
void HS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif  // HS_H

