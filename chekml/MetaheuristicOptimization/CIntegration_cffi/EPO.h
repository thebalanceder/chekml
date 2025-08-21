#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define INITIAL_F 2.0
#define INITIAL_L 1.5
#define INITIAL_M 0.5
#define ADAPTATION_INTERVAL 10
#define STRATEGY_COUNT 3

// ⚙️ Strategy Identifiers
#define STRATEGY_LINEAR 0
#define STRATEGY_EXPONENTIAL 1
#define STRATEGY_CHAOTIC 2

// 🌊 EPO Algorithm Phases
void initialize_penguins(Optimizer *opt);
void compute_huddle_boundary(double *T_prime, double *R, int t, int max_iter);
void adapt_parameters(double *f, double *l, double *m_param, int strategy, double t_norm, int t);
void update_strategy_probabilities(double *strategy_probs, double *strategy_success);
void huddle_movement(Optimizer *opt, int index, int t, ObjectiveFunction objective_function, double *workspace);
void EPO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif
