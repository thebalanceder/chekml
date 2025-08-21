#ifndef DISO_WITH_RCF_H
#define DISO_WITH_RCF_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define DIVERSION_FACTOR 0.3
#define FLOW_ADJUSTMENT 0.2
#define WATER_DENSITY 1.35
#define FLUID_DISTRIBUTION 0.46
#define CENTRIFUGAL_RESISTANCE 1.2
#define BOTTLENECK_RATIO 0.68
#define ELIMINATION_RATIO 0.23

// ⚙️ Hydrodynamic Constants
#define HRO 1.2
#define HRI 7.2
#define HGO 1.3
#define HGI 0.82
#define CFR_FACTOR 9.435

// 🌊 DISO Algorithm Phases
void diversion_phase(Optimizer *opt);
void spiral_motion_update(Optimizer *opt, int t);
void local_development_phase(Optimizer *opt);
void elimination_phase(Optimizer *opt);

// 🚀 Optimization Execution
void DISO_optimize(Optimizer *opt, double (*objective_function)(double *));

// 🛑 **Fix 3:** Ensure boundaries are enforced
void enforce_bound_constraints(Optimizer *opt);
int compare_fitness_diso(const void *a, const void *b);

#ifdef __cplusplus
}
#endif

#endif // DISO_WITH_RCF_H

