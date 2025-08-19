#ifndef CBO_H
#define CBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameter
#define CBO_ALPHA 0.1f

// Optimization execution
void CBO_optimize(Optimizer* opt, double (*objective_function)(double*));

#ifdef __cplusplus
}
#endif

#endif // CBO_H
