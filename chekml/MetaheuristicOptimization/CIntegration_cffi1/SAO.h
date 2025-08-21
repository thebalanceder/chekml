#ifndef SAO_H
#define SAO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization Parameters
#define NUM_POP 50
#define SAO_MAX_ITER 1000
#define LOW -5.0
#define UP 5.0
#define DF_MIN 0.35
#define DF_MAX 0.6
#define NUM_ELITES 4
#define BROWNIAN_VARIANCE 0.5

// ⚙️ Algorithm Constants
#define EULER 2.718281828459045

// 🌨️ SAO Algorithm Phases
void initialize_population_sao(Optimizer *opt);
void brownian_motion(double **motion, int num_pop, int dim);
void calculate_centroid(Optimizer *opt, double *centroid);
void select_elite(Optimizer *opt, double *elite, int *best_idx);
void covariance_matrix_learning(Optimizer *opt, double **QQ);
void historical_boundary_adjustment(double *position, int dim);
void random_centroid_reverse_learning(Optimizer *opt, double **reverse_pop);
double calculate_snow_ablation_rate(int iter);
void enforce_sao_bound_constraints(Optimizer *opt);

// 🚀 Optimization Execution
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SAO_H
