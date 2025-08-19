#ifndef SFO_H
#define SFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define TV1 0.5
#define TV2 0.3
#define P0 0.25
#define K 0.4

// 🌊 SFO Algorithm Phases
void large_scale_search(Optimizer *opt, int idx, double r1);
void raid(Optimizer *opt, int idx, double r1, double w);
void transition_phase(Optimizer *opt, int idx, double r2, double w, double instruction);
void arrest_rescue(Optimizer *opt);
void unmanned_search(Optimizer *opt, int t);

// 🚀 Optimization Execution
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SFO_H
