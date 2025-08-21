#ifndef HPO_H
#define HPO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CONSTRICTION_COEFF 0.1
#define C_PARAM_MAX 0.98

// HPO Algorithm Phases
void hpo_update_positions(Optimizer *opt, int iter);

// Optimization Execution
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HPO_H
