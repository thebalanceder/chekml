#ifndef MA_H
#define MA_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MA_MUTATION_RATE 0.1
#define MA_CROSSOVER_RATE 0.8
#define MA_LOCAL_SEARCH_RATE 0.1
#define MA_LOCAL_SEARCH_ITERS 100
#define MA_MUTATION_STDDEV 0.1

void MA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif  // MA_H
