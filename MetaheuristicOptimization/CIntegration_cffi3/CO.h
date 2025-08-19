#ifndef CO_H
#define CO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization Parameters
#define MIN_EGGS 2
#define MAX_EGGS 4
#define MAX_CUCKOOS 10
#define MAX_EGGS_PER_CUCKOO 4
#define RADIUS_COEFF 5.0f
#define MOTION_COEFF 9.0f
#define KNN_CLUSTER_NUM 1
#define VARIANCE_THRESHOLD 1e-13
#define PI 3.14159265358979323846f

// Optimization Execution
void CO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CO_H
