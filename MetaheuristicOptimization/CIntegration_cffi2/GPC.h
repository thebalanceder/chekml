#ifndef GPC_H
#define GPC_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>

// ⚙️ Physical Constants
#define G 9.8                     // Gravity constant
#define THETA (14.0 * M_PI / 180) // Ramp angle in radians
#define MU_MIN 0.3                // Minimum friction coefficient
#define MU_MAX 0.5                // Maximum friction coefficient
#define V_MIN 0.1                 // Minimum velocity
#define V_MAX 2.0                 // Maximum velocity

// 📌 Optimization Algorithm Phases
double random_uniform(double min, double max);
void compute_movement(double velocity, double friction, double* d, double* x);
void update_population(Optimizer* opt);
void evaluate_population(Optimizer* opt, ObjectiveFunction objective_function);

// 🚀 Main Optimization Function
void GPC_optimize(Optimizer* opt, ObjectiveFunction objective_function);

void enforce_bound_constraints(Optimizer* opt);

#ifdef __cplusplus
}
#endif

#endif // GPC_H

