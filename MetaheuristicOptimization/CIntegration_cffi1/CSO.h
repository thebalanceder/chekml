#ifndef CSO_H
#define CSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define UPDATE_FREQ 10
#define ROOSTER_RATIO 0.15
#define HEN_RATIO 0.7
#define MOTHER_RATIO 0.5

// Forward declaration of CSO_Buffers
typedef struct CSO_Buffers CSO_Buffers;

// CSO Algorithm Phases
void initialize_population_cso(Optimizer *opt, double (*objective_function)(double *), CSO_Buffers *buffers);
void CSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CSO_H
