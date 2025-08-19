#ifndef WPA_H
#define WPA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define R1_MIN 0.0
#define R1_MAX 2.0
#define R2_MIN 0.0
#define R2_MAX 1.0
#define R3_MIN 0.0
#define R3_MAX 2.0
#define K_INITIAL 1.0
#define F_MIN -5.0
#define F_MAX 5.0
#define C_MIN -5.0
#define C_MAX 5.0
#define STAGNATION_THRESHOLD 3

// ⚙️ Algorithm Constants
#ifndef PI
#define PI 3.14159265358979323846
#endif

double rand_double(double min, double max);

// 🌱 WWPA Algorithm Phases
void wpa_exploration_phase(Optimizer *opt, int t, double K, int *stagnation_counts, double (*objective_function)(double *));
void wpa_exploitation_phase(Optimizer *opt, int t, double K, int *stagnation_counts, double (*objective_function)(double *));
double wpa_update_k(int t, int max_iter, double f);

// 🚀 Optimization Execution
void WPA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WPA_H
