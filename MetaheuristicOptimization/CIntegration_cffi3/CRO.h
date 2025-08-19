#ifndef CRO_H
#define CRO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CRO_NUM_REEFS 10
#define CRO_ALPHA 0.1f

// Optimization execution
void CRO_optimize(Optimizer* opt, double (*objective_function)(double*));

#ifdef __cplusplus
}
#endif

#endif // CRO_H
