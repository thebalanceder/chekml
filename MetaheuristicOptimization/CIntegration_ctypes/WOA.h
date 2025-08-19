#ifndef WOA_H
#define WOA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define WOA_A_INITIAL 2.0
#define WOA_A2_INITIAL -1.0
#define WOA_A2_FINAL -2.0
#define WOA_B 1.0

// WOA Algorithm Phases
void initialize_positions(Optimizer *opt);
void update_leader(Optimizer *opt, double (*objective_function)(double *));
void update_positions_woa(Optimizer *opt, int t);

// Optimization Execution
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WOA_H
