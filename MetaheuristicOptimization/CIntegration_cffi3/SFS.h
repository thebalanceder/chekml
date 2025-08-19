#ifndef SFS_H
#define SFS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// SFS Parameters
#define SFS_WALK_PROB 0.8f
#define SFS_MAX_DIFFUSION 4
#define SFS_MAX_DIM 100
#define SFS_EARLY_STOP_THRESHOLD 1e-6f

// Main Optimization Function
void SFS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SFS_H
