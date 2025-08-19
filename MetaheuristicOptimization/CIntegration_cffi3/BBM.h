#ifndef BBM_H
#define BBM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define QUEEN_FACTOR 0.3f
#define DRONE_SELECTION 0.2f
#define WORKER_IMPROVEMENT 1.35f
#define BROOD_DISTRIBUTION 0.46f
#define MATING_RESISTANCE 1.2f
#define REPLACEMENT_RATIO 0.23f
#define BBM_CFR_FACTOR 9.435f
#define BLEND_ALPHA 0.5f

// Function prototypes
void BBM_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // BBM_H
