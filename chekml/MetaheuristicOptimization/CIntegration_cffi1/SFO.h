#ifndef SFO_H
#define SFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define SFO_ALPHA 0.1f // Step size for sailfish movement
#define SFO_AP_MAX 1.0 // Initial attack power
#define SFO_AP_MIN 0.01 // Minimum attack power
#define SFO_SARDINE_RATIO 0.5 // Fraction of population as sardines
#define ENABLE_DEBUG 0 // Toggle debugging output (0 for max speed)

// 🚀 SFO Algorithm Phases
void initialize_sardines(Optimizer *opt, Solution *sardines, int sardine_count);
void update_sailfish(Optimizer *opt, Solution *sardines, int sardine_count, double ap);
void update_sardines(Optimizer *opt, Solution *sardines, int sardine_count);
void update_best_solution_sfo(Optimizer *opt, Solution *sardines, int sardine_count, double (*objective_function)(double *));

// 🚀 Optimization Execution
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SFO_H
