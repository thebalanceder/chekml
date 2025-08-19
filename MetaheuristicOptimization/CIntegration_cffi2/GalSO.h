#ifndef GALSO_H
#define GALSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "generaloptimizer.h"

// 🔧 Galactic Swarm Optimization Parameters
#define DEFAULT_POP_SIZE 5
#define DEFAULT_SUBPOP 10
#define DEFAULT_EPOCH_NUMBER 5
#define DEFAULT_ITERATION1 10
#define DEFAULT_ITERATION2 10
#define DEFAULT_TRIALS 5
#define C1_MAX 2.05
#define C2_MAX 2.05
#define C3_MAX 2.05
#define C4_MAX 2.05

// 🚀 Optimization Function
void GalSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GALSO_H
