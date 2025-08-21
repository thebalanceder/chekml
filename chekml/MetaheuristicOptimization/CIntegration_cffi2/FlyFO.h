#ifndef FLYFO_H
#define FLYFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define DELTASO_0 0.2
#define DELTASO_1 0.4
#define DELTASO_2 0.6
#define ALPHA_PARAM_0 1.0
#define ALPHA_PARAM_1 1.5
#define ALPHA_PARAM_2 1.9
#define PA_PARAM_0 0.5
#define PA_PARAM_1 0.85
#define PA_PARAM_2 0.99
#define SURVIVAL_LIST_RATIO 0.25
#define DEATH_PROB_FACTOR 1.0

// ⚙️ Algorithm Constants
#define MAX_EVALS_DEFAULT 100
#define POPULATION_SIZE_FACTOR 10
#define SQRT_DIM_FACTOR 2

// 🦇 FFO Context for Additional State
typedef struct {
    double worst_fitness;
    Solution *survival_list;
    int survival_list_count;
    double *past_fitness;  // Array to store past fitness for each solution
} FlyFOContext;

// 🦇 FFO Algorithm Phases
void initialize_population_flyfo(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *));
void fuzzy_self_tuning(Optimizer *opt, FlyFOContext *ctx, int i, double deltamax, double *alpha, double *pa);
void update_position(Optimizer *opt, FlyFOContext *ctx, int i, double alpha, double pa, double *deltas, double (*objective_function)(double *));
void update_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position, double new_fitness, int surv_list_size);
void replace_with_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position, int surv_list_size);
void crossover(Optimizer *opt, int parent1, int parent2, double *off1, double *off2, double *randoms, int dim);
void suffocation_phase(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *), int surv_list_size);

// 🚀 Optimization Execution
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FLYFO_H
