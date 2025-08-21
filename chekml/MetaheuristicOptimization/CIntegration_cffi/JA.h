#ifndef JAGUAR_ALGORITHM_H
#define JAGUAR_ALGORITHM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CRUISING_PROBABILITY 0.8
#define CRUISING_DISTANCE 0.1
#define ALPHA 0.1

// Jaguar Algorithm Phases
void ja_cruising_phase(Optimizer *opt, int index, int iteration, double *direction);
void ja_random_walk_phase(Optimizer *opt, int index, double *direction);
void JA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // JAGUAR_ALGORITHM_H
