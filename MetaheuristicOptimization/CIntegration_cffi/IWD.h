#ifndef IWD_H
#define IWD_H

#pragma once

#ifdef __cplusplus
extern "C" {
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
void initialize_iwd_population(Optimizer *restrict opt);
void move_water_drop(Optimizer *restrict opt, int iwd_idx, int *restrict visited, int *restrict visited_count, 
                     double *restrict soil_amount, double *restrict soil, double *restrict hud, 
                     char *restrict visited_flags, int *restrict valid_nodes, double *restrict probabilities);
void update_iteration_best(Optimizer *restrict opt, int *restrict visited, int visited_count, 
                          double soil_amount, int iwd_idx, double *restrict soil);

// Main Optimization Function
void IWD_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict));

#ifdef __cplusplus
}
#endif

#endif // IWD_H
