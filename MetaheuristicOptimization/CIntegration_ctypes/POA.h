#ifndef POA_H
#define POA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define STEP_SIZE 0.1

// 🌱 POA Algorithm Phases
void initialize_population_poa(Optimizer *opt);
void evaluate_population_poa(Optimizer *opt, double (*objective_function)(double *));
void update_positions_poa(Optimizer *opt);

// 🚀 Optimization Execution
void POA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // POA_H
