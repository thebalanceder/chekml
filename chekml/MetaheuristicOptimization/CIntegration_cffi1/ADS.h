#ifndef ADS_H
#define ADS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "generaloptimizer.h"

// Adaptive Dimension Search Parameters
#define MAX_COLLOCATION_POINTS 1000
#define EPSILON_C 0.01
#define GAMMA_C 0.01
#define ETA_C 0.01

void ADS_optimize(Optimizer *opt, double (*performance_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ADS_H
