#ifndef ACO_H
#define ACO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 ACO Parameters
#define ACO_MAX_ITER 300
#define ACO_N_ANT 40
#define ACO_Q 1.0
#define ACO_TAU0 0.1
#define ACO_ALPHA 1.0
#define ACO_BETA 0.02
#define ACO_RHO 0.1
#define ACO_N_BINS 10

// 🐜 ACO Algorithm Functions
double rand_double_aco(double min, double max);
void construct_solutions(Optimizer *opt, double (*objective_function)(double *), double **tau, double **bins, int **tours, double *prob);
void update_pheromones(Optimizer *opt, double **tau, int **tours);
void evaporate_pheromones(double **tau, int dim);
void ACO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ACO_H
