#ifndef CO_H
#define CO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define MIN_EGGS 2
#define MAX_EGGS 4
#define MAX_CUCKOOS 10
#define RADIUS_COEFF 5.0
#define MOTION_COEFF 9.0
#define KNN_CLUSTER_NUM 1
#define VARIANCE_THRESHOLD 1e-13
#define PI_CO 3.14159265358979323846

// 🐦 Cuckoo Optimization Phases
void initialize_cuckoos(Optimizer *opt);
void lay_eggs(Optimizer *opt, double **egg_positions, int *num_eggs, int *total_eggs);
void select_best_cuckoos(Optimizer *opt, double **positions, double *fitness, int num_positions);
int cluster_and_migrate(Optimizer *opt);

// 🚀 Optimization Execution
void CO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif //CO_H