#ifndef MSO_H
#define MSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define P_EXPLORE 0.2
#define MAX_P_EXPLORE 0.8
#define MIN_P_EXPLORE 0.1
#define PERTURBATION_SCALE 10.0

// 🐒 MSA Algorithm Functions
double rand_normal_mso(double mean, double stddev);
void update_positions_mso(Optimizer *opt, int iter);
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // MSO_H
