#ifndef FPA_H
#define FPA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure core utilities
#include <stdio.h>   // ✅ For minimal logging
#include <immintrin.h>  // ✅ For SSE/AVX intrinsics
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define FPA_SWITCH_PROB 0.8
#define FPA_LEVY_STEP_SCALE 0.01
#define FPA_LEVY_BETA 1.5
#define FPA_LEVY_SIGMA 0.6966  // Precomputed for beta = 1.5
#define FPA_INV_LEVY_BETA (1.0 / 1.5)  // Precomputed 1/beta
#define FPA_MAX_DIM 128  // Max dimension for stack-based buffers
#define FPA_LEVY_LUT_SIZE 256  // Lookup table size for Lévy steps

// ⚙️ FPA Constants
#define FPA_POPULATION_SIZE_DEFAULT 20
#define FPA_MAX_ITER_DEFAULT 5000

// 🌸 FPA Algorithm Phases
void fpa_initialize_flowers(Optimizer *restrict opt);
void fpa_global_pollination_phase(Optimizer *restrict opt, double *restrict step_buffer);
void fpa_local_pollination_phase(Optimizer *restrict opt);

// 🚀 Optimization Execution
void FPA_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict));

#ifdef __cplusplus
}
#endif

#endif // FPA_H
