#ifndef FPA_H
#define FPA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define FPA_SWITCH_PROB 0.8
#define FPA_LEVY_STEP_SCALE 0.01
#define FPA_LEVY_BETA 1.5
#define FPA_LEVY_SIGMA 0.6966  // Precomputed for beta = 1.5
#define FPA_INV_LEVY_BETA (1.0 / 1.5)  // Precomputed 1/beta for speed

// ⚙️ FPA Constants
#define FPA_POPULATION_SIZE_DEFAULT 20
#define FPA_MAX_ITER_DEFAULT 5000

// 🌸 FPA Algorithm Phases
void fpa_initialize_flowers(Optimizer *opt);
void fpa_global_pollination_phase(Optimizer *opt, double *step_buffer);
void fpa_local_pollination_phase(Optimizer *opt);

// 🚀 Optimization Execution
void FPA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FPA_H
