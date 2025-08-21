#ifndef EHO_H
#define EHO_H
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include <CL/cl.h>
#include "generaloptimizer.h"
// Optimization parameters
#define EHO_ALPHA 0.5f
#define EHO_BETA 0.1f
#define EHO_KEEP 2
#define EHO_NUM_CLANS 5
// Optimization execution
void EHO_optimize(Optimizer* opt, ObjectiveFunction objective_function);
#ifdef __cplusplus
}
#endif
#endif // EHO_H
