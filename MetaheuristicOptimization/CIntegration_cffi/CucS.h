#ifndef CUCS_H
#define CUCS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CS_POPULATION_SIZE 25
#define CS_MAX_ITER 100
#define CS_PA 0.25
#define CS_BETA 1.5
#define CS_STEP_SCALE 0.01

// Precomputed Levy flight constants
#define CS_PI 3.141592653589793
#define CS_GAMMA_BETA 1.225
#define CS_GAMMA_HALF_BETA 0.886
#define CS_SIGMA 0.696066 // Precomputed sigma for Levy flight

// Function declarations
void initialize_nests(Optimizer *restrict opt);
void evaluate_nests(Optimizer *restrict opt, double (*objective_function)(double *));
void get_cuckoos(Optimizer *restrict opt);
void empty_nests(Optimizer *restrict opt);
void CucS_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif
