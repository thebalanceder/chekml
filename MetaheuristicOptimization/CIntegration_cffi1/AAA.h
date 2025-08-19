#ifndef AAA_H
#define AAA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define STEP_SIZE 0.1

// AAA Algorithm Phases
void initialize_population_aaa(Optimizer *opt);
void evaluate_population_aaa(Optimizer *opt, double (*objective_function)(double *));

// Optimization Execution
void AAA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // AAA_H
