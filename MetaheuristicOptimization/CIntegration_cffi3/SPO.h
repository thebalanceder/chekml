#ifndef SPO_H
#define SPO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// SPO-specific function prototypes
void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void enforce_bound_constraints(Optimizer* opt);

#ifdef __cplusplus
}
#endif

#endif // SPO_H
