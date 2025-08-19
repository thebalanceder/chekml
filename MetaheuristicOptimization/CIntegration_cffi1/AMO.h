#ifndef AMO_H
#define AMO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include "generaloptimizer.h"

// 🔧 Optimization Parameters
#define NEIGHBORHOOD_SIZE 5
#define POPULATION_SCALE_FACTOR 1.0
#define MIGRATION_PROBABILITY_FACTOR 0.5
#define FIXED_DIM 2 // Fixed dimension for unrolling and SIMD
#define CACHE_LINE_SIZE 64 // Bytes

// ⚙️ Algorithm Constants
#define NEIGHBORHOOD_RADIUS 2
#define FITNESS_SCALING 1.0
#define NORMAL_TABLE_SIZE 1024 // Precomputed Gaussian variates

// Aligned memory allocation
#define ALIGNED_MALLOC(size) _mm_malloc(size, CACHE_LINE_SIZE)
#define ALIGNED_FREE(ptr) _mm_free(ptr)

// 🌍 AMO Algorithm Phases
void initialize_population_amo(Optimizer *opt);
void neighborhood_learning_phase(Optimizer *opt);
void global_migration_phase(Optimizer *opt);

// 🚀 Optimization Execution
void AMO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // AMO_H
