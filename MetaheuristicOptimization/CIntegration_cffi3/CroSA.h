#ifndef CROSA_H
#define CROSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define AWARENESS_PROBABILITY 0.1
#define FLIGHT_LENGTH 2.0

// 🚀 Optimization Execution
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CROSA_H
