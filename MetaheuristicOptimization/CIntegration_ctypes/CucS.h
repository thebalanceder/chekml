#ifndef CUCS_H
#define CUCS_H

#pragma once  // Ensures header is included only once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // For malloc/free and rand
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define CS_POPULATION_SIZE 25
#define CS_MAX_ITER 1000
#define CS_PA 0.25  // Discovery probability
#define CS_BETA 1.5  // Levy flight exponent
#define CS_STEP_SCALE 0.01  // Step size scaling factor

// Levy flight constants
#define CS_PI 3.141592653589793
#define CS_GAMMA_BETA 1.225  // Approximation for gamma(1 + beta)
#define CS_GAMMA_HALF_BETA 0.886  // Approximation for gamma((1 + beta) / 2)

// Cuckoo Search algorithm phases
void initialize_nests(Optimizer *opt);
void evaluate_nests(Optimizer *opt, double (*objective_function)(double *));
void get_cuckoos(Optimizer *opt);
void empty_nests(Optimizer *opt);
void CucS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CUCKOO_SEARCH_H
