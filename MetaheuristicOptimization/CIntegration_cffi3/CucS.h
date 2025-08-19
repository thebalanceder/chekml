#ifndef CUCS_H
#define CUCS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CS_POPULATION_SIZE 25
#define CS_MAX_ITER 100
#define CS_PA 0.25f
#define CS_BETA 1.5f
#define CS_STEP_SCALE 0.01f

// Precomputed Levy flight constants
#define CS_PI 3.141592653589793f
#define CS_GAMMA_BETA 1.225f
#define CS_GAMMA_HALF_BETA 0.886f
#define CS_SIGMA 0.696066f // Precomputed sigma for Levy flight

// Function declaration
void CucS_optimize(Optimizer *restrict opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif
