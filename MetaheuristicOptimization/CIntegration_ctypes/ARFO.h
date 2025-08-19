#ifndef ARFO_H
#define ARFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define BRANCHING_THRESHOLD 0.6
#define MAX_BRANCHING 5
#define MIN_BRANCHING 1
#define INITIAL_STD 1.0
#define FINAL_STD 0.01
#define MAX_ELONGATION 0.1
#define LOCAL_INERTIA 0.5
#define ELIMINATION_PERCENTILE 10.0

// ⚙️ Algorithm Constants
#define AUXIN_NORMALIZATION_FACTOR 1.0
#define MEDIAN_THRESHOLD 0.5

// 🌱 ARFO Algorithm Phases
void regrowth_phase(Optimizer *opt);
void branching_phase(Optimizer *opt, int t);
void lateral_growth_phase(Optimizer *opt);
void elimination_phase_arfo(Optimizer *opt);
void replenish_phase(Optimizer *opt);

// 🚀 Optimization Execution
void ARFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ARFO_H
