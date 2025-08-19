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
#define NUM_PARENTS_RATIO 0.3
#define NUM_OFFSPRINGS 3
#define MAX_FLA_ITER 5
#define SFL_STEP_SIZE 2.0

// 🔧 SFLA Context for Objective Function
typedef struct {
    double (*objective_function)(double *);
} SFLContext;

// ⚙️ Utility Functions
void sort_population(Optimizer *opt, SFLContext *ctx);

// 🌊 SFLA Algorithm Phases
void initialize_population_sfl(Optimizer *opt);
int is_in_range(const double *position, const double *bounds, int dim);
void run_fla(Optimizer *opt, Solution *memeplex, int memeplex_size, SFLContext *ctx, double *precomputed_P);

// 🚀 Optimization Execution
void SFL_optimize(void *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SFL_H
