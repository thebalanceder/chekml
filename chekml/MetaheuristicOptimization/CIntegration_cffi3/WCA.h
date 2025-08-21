#ifndef WCA_H
#define WCA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// WCA Parameters
#define WCA_PERTURBATION_FACTOR 0.1f
#define WCA_LSAR_MAX_ITER 1000
#define WCA_MAX_DIM 100
#define WCA_EARLY_STOP_THRESHOLD 1e-6f
#define WCA_CONVERGENCE_THRESHOLD 1e-8f

// Main Optimization Function
void WCA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // WCA_H
