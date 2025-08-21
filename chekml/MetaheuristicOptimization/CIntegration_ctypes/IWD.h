#ifndef IWD_H
#define IWD_H

#pragma once

#ifdef __cplusplus
extern "C" { // ✅ Improves header inclusion efficiency
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define A_S 1.0
#define B_S 0.01
#define C_S 1.0
#define A_V 1.0
#define B_V 0.01
#define C_V 1.0
#define INIT_VEL 200.0
#define P_N 0.9
#define P_IWD 0.9
#define INITIAL_SOIL 10000.0
#define EPSILON_S 0.0001

// IWD Algorithm Phases
void initialize_iwd_population(Optimizer *opt);
void move_water_drop(Optimizer *opt, int iwd_idx, int *visited, int *visited_count, double *soil_amount, double *soil, double *hud);
void update_iteration_best(Optimizer *opt, int *visited, int visited_count, double soil_amount, int iwd_idx, double *soil);

// Main Optimization Function
void IWD_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // IWD_H
