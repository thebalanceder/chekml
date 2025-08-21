#ifndef MA_H
#define MA_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

void MA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif  // MA_H
