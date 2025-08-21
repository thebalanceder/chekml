#ifndef EA_H
#define EA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// Parameters for EA
#define EA_POPULATION_SIZE 100
#define EA_GENERATIONS 100
#define EA_DIM 2
#define EA_MU 15
#define EA_MUM 20
#define EA_CROSSOVER_PROB 0.9

// Function declarations
void EA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif /* EA_H */

