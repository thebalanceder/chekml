#ifndef BDFO_H
#define BDFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure rand/RAND_MAX work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define EXPLORATION_FACTOR 0.5
#define ADJUSTMENT_RATE 0.3
#define ELIMINATION_RATIO 0.23
#define BDFO_PERTURBATION_SCALE 0.1  // Renamed to avoid conflict with MSO.h
#define EXPLORATION_PROB 0.3

// ⚙️ Algorithm-specific constants
#define SEGMENT_FACTOR 0.5  // Split solutions into two segments (half_dim = dim * SEGMENT_FACTOR)

// 🐬 BDFO Algorithm Phases
void bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *));
void bdfo_exploration_phase(Optimizer *opt);  // Renamed to avoid conflict with HHO.h
void elimination_phase_bdfo(Optimizer *opt);

// 🚀 Optimization Execution
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BDFO_H
