#ifndef JAGUAR_ALGORITHM_H
#define JAGUAR_ALGORITHM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// Optimization parameters
#define JA_CRUISING_PROBABILITY 0.8
#define JA_CRUISING_DISTANCE 0.1
#define JA_ALPHA 0.1

// Jaguar Algorithm API
void JA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // JAGUAR_ALGORITHM_H

