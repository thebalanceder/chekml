#ifndef DE_H
#define DE_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// Differential Evolution parameters
#define DE_F 0.5
#define DE_CR 0.9

// Strategy options
#define STRATEGY_RAND_1_BIN 0
#define STRATEGY_BEST_1_BIN 1
#define STRATEGY_RAND_TO_BEST_1 2

#define STRATEGY 0

void DE_optimize(Optimizer* opt, ObjectiveFunction objective_function, int strategy);

#ifdef __cplusplus
}
#endif

#endif /* DE_H */

