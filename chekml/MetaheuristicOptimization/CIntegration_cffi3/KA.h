#ifndef KA_H
#define KA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// KA parameters
#define KA_P1 0.2f
#define KA_P2 0.5f
#define KA_S_MAX 4
#define KA_LOCAL_WORK_SIZE 64

// Function declarations
void KA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // KA_H
