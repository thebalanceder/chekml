#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>  // ✅ For fixed-width integers
#include <immintrin.h>  // ✅ For SSE/AVX intrinsics
#include <stdio.h>  // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization Parameters
#define MIN_EGGS 2
#define MAX_EGGS 4
#define MAX_CUCKOOS 10
#define MAX_EGGS_PER_CUCKOO 4
#define RADIUS_COEFF 5.0
#define MOTION_COEFF 9.0
#define VARIANCE_THRESHOLD 1e-13
#define PI 3.14159265358979323846
#define MAX_POP 50  // Max population size
#define MAX_DIM 10  // Max dimensions
#define MAX_TOTAL_EGGS (MAX_POP * MAX_EGGS_PER_CUCKOO)

// 🌊 Cuckoo Optimization Execution
void CO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif
