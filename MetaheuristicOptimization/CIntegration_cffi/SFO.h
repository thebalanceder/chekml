#ifndef SFO_H
#define SFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>  // ✅ For uint32_t
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#ifdef _OPENMP
#include <omp.h>     // ✅ For parallelization
#endif
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define TV1 0.5
#define TV2 0.3
#define P0 0.25
#define K 0.4

// 🌊 SFO Algorithm Phases
typedef struct { uint32_t x, y, z, w; } SFOXorshiftState; // Unique RNG state for SFO
void sfo_exploration_phase(Optimizer *opt, int idx, double r1, double w, double *temp_pos, SFOXorshiftState *state);
void transition_phase(Optimizer *opt, int idx, double r2, double w, double instruction, double *temp_pos, SFOXorshiftState *state);
void arrest_rescue(Optimizer *opt, double *temp_pos, SFOXorshiftState *state);
void unmanned_search(Optimizer *opt, int t, double *temp_pos, double *temp_vec, SFOXorshiftState *state);

// 🚀 Optimization Execution
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *));

// ✅ Note: Compile with -O3 -ffast-math -march=native for best performance
// ✅ If using OpenMP, add -fopenmp

#ifdef __cplusplus
}
#endif

#endif // SFO_H
