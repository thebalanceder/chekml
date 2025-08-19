#ifndef CFO_H
#define CFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameter
#define CFO_ALPHA 0.1f

// Optimization execution
void CFO_optimize(Optimizer* opt, double (*objective_function)(double*));

#ifdef __cplusplus
}
#endif

#endif // CFO_H
