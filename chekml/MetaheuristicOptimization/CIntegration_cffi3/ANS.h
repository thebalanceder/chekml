#ifndef ANS_H
#define ANS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// ANS Parameters
#define ANS_NUM_NEIGHBORHOODS 5
#define ANS_MUTATION_RATE 0.1f
#define VERBOSE 1

// ANS API
void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // ANS_H
