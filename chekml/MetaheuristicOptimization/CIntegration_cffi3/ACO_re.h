#ifndef ACO_H
#define ACO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"
#include <CL/cl.h>

// 🔧 ACO Parameters
#define ACO_MAX_ITER 300
#define ACO_N_ANT 40
#define ACO_Q 1.0f
#define ACO_TAU0 0.1f
#define ACO_ALPHA 1.0f
#define ACO_BETA 0.02f
#define ACO_RHO 0.1f
#define ACO_N_BINS 10

// 🐜 ACO Algorithm Functions
void ACO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ACO_H
