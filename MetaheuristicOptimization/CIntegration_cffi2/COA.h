#ifndef COA_H
#define COA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure rand() and RAND_MAX are available
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define N_PACKS 10  // Adjusted to match population_size (50) / N_COYOTES_PER_PACK (5)
#define N_COYOTES_PER_PACK 5
#define MAX_NFEVAL 20000
#define P_LEAVE 0.01
#define PS 1.0
#define SCALE_FACTOR_DIVERSITY_THRESHOLD 0.1
#define SCALE_FACTOR_HIGH 1.5
#define SCALE_FACTOR_LOW 1.0

// ⚙️ Algorithm Constants
#define AGE_INCREMENT 1.0

// 🐺 COA Algorithm Phases
void initialize_coyotes(Optimizer *opt, double (*objective_function)(double *));
void update_pack(Optimizer *opt, int pack_idx, double (*objective_function)(double *));
void pack_exchange(Optimizer *opt);
void limit_bounds(Optimizer *opt, double *X, double *X_clipped, int size);

// 🚀 Optimization Execution
void COA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // COA_H
