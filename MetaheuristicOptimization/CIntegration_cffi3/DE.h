#ifndef DE_H
#define DE_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Differential Evolution parameters
#define DE_F 0.5f    // Scaling factor
#define DE_CR 0.9f   // Crossover probability
#define DE_TOL 1e-6f // Convergence tolerance

// Define the DE strategy (select one by setting the value)
// 0: STRATEGY_RAND_1_BIN (v = pop[r1] + DE_F * (pop[r2] - pop[r3]))
// 1: STRATEGY_BEST_1_BIN (v = best + DE_F * (pop[r1] - pop[r2]))
// 2: STRATEGY_RAND_TO_BEST_1 (v = pop[r1] + 0.5 * (best - pop[r1]) + DE_F * (pop[r2] - pop[r3]))
#define DE_STRATEGY 0

void DE_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif /* DE_H */
