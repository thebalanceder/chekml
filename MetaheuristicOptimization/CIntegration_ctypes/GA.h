// GA.h
#ifndef GA_H
#define GA_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GA_NUM_POPULATIONS 1  // Number of outer population sets (like NumPop in MATLAB)
#define GA_VERBOSITY 1        // Set to 0 to disable console logging

void GA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // GA_H

