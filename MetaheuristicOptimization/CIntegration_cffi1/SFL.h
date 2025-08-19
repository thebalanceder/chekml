#ifndef SFL_H
#define SFL_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 SFLA Parameters
#define MEMEPLEX_SIZE 10
#define NUM_MEMEPLEXES 5
#define NUM_PARENTS 3
#define NUM_OFFSPRINGS 3
#define MAX_FLA_ITER 5
#define SFL_STEP_SIZE 2.0
#ifndef POPULATION_SIZE
#define POPULATION_SIZE (MEMEPLEX_SIZE * NUM_MEMEPLEXES)
#endif

// 🔧 SFLA Context
typedef struct {
    double (*objective_function)(double *);
} SFLContext;

// 🚀 Optimization Execution
void SFL_optimize(void *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SFL_H
