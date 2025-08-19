#ifndef CBO_H
#define CBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define ALPHA 0.1

// 🚀 CBO Algorithm Phases
void collision_phase(Optimizer *opt);
void update_best_solution(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void CBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CBO_H