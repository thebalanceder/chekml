#ifndef HHO_H
#define HHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h> // AVX intrinsics
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 HHO Parameters
#define HHO_BETA 1.5  // Levy flight parameter
#define ENERGY_FACTOR 2.0  // Initial energy factor
#define LEVY_SIGMA 0.696568676784  // Precomputed for beta=1.5

// 🌟 HHO Algorithm Phases
void exploration_phase(Optimizer *opt, double (*objective_function)(double *));
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *));
void levy_flight(double *step, int dim, uint32_t *rng_state);

// 🚀 Optimization Execution
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *));

// 🛠 Utility Functions
static inline uint32_t pcg32(uint32_t *state);
static inline double hho_rand_double(double min, double max, uint32_t *rng_state);

#ifdef __cplusplus
}
#endif

#endif // HHO_H
