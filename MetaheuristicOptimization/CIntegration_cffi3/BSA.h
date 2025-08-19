#ifndef BSA_H
#define BSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameter
#define BSA_DIM_RATE 0.5f

// Function declaration
void BSA_optimize(Optimizer* opt, double (*objective_function)(double*));

#ifdef __cplusplus
}
#endif

#endif /* BSA_H */
