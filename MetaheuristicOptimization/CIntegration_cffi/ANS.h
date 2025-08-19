#ifndef ANS_H
#define ANS_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ANS_NUM_NEIGHBORHOODS 5
#define ANS_MUTATION_RATE 0.1
#define VERBOSE 1  // Set to 0 to disable progress printing

void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif  // ANS_H
