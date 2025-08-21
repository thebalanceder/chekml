#ifndef ABC_H
#define ABC_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// 🔧 ABC Algorithm Parameters
#define ABC_POPULATION_SIZE 100
#define ABC_MAX_ITER 200
#define ABC_ONLOOKER_RATIO 1.0
#define ABC_TRIAL_LIMIT_FACTOR 0.6
#define ABC_ACCELERATION_BOUND 1.0

// 🚀 Optimization Execution
void ABC_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ABC_H
