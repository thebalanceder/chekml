#ifndef DEA_H
#define DEA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// DEA Optimization Parameters
#define DEA_POPULATION_SIZE 50
#define DEA_MAX_LOOPS 100
#define DEA_CONVERGENCE_POWER 1.0f
#define DEA_EFFECTIVE_RADIUS_FACTOR 0.25f
#define DEA_PROBABILITY_THRESHOLD 0.1f
#define DEA_ALTERNATIVES_PER_DIM 100

// Function declaration
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DEA_H
