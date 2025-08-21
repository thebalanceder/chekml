#ifndef BSA_H
#define BSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Parameter for dimension rate
#define BSA_DIM_RATE 0.5

// Function declarations
double rand_double(double min, double max);
void BSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif /* BSA_H */