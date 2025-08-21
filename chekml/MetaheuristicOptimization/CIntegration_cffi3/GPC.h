#ifndef GPC_H
#define GPC_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// ⚙️ Physical Constants
#define G 9.8f                    // Gravity constant
#define THETA (14.0f * M_PI / 180.0f) // Ramp angle in radians
#define MU_MIN 0.3f               // Minimum friction coefficient
#define MU_MAX 0.5f               // Maximum friction coefficient
#define V_MIN 0.1f                // Minimum velocity
#define V_MAX 2.0f                // Maximum velocity

// 📌 GPC-specific parameters
#define GPC_POPULATION_SIZE 256
#define GPC_LOCAL_WORK_SIZE 64

// 🚀 Main Optimization Function
void GPC_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // GPC_H
