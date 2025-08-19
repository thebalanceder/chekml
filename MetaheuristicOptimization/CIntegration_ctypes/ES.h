#ifndef EAGLE_STRATEGY_H
#define EAGLE_STRATEGY_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// 🧠 Cognitive and Social Coefficients
#define C1 2.0
#define C2 2.0

// 🪶 Inertia Weight
#define W_MAX 0.9
#define W_MIN 0.4

// 🦅 Lévy flight parameters
#define LEVY_BETA 1.5
#define LEVY_STEP_SCALE 0.1
#define LEVY_PROBABILITY 0.2

// Random generator
double rand_double_es(double min, double max);

// Main function
void ES_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif  // EAGLE_STRATEGY_H

