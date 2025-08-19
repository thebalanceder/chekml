#ifndef HBO_H
#define HBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// HBO-specific parameters
#define HBO_POPULATION_SIZE 256
#define HBO_LOCAL_WORK_SIZE 64

// Main Optimization Execution
void HBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HBO_H
