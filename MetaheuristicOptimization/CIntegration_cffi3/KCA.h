#ifndef KCA_H
#define KCA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define PROBABILITY_THRESHOLD 0.5f
#define HALF_POPULATION_FACTOR 0.5f
#define KEY_INIT_PROB 0.5f
#define FITNESS_SCALE 1.0f
#define BITS_PER_DIM 10
#define SHUBERT_MIN -5.12f
#define SHUBERT_MAX 5.12f

// ⚙️ KCA Constants
#define KCA_MIN_KEY_LENGTH 1
#define KCA_MAX_KEY_LENGTH 1000
#define KCA_MIN_POP_SIZE 2
#define KCA_MAX_POP_SIZE 1000
#define KCA_DEFAULT_MAX_ITER 100

// 🌊 KCA Algorithm Function
void KCA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

// Precompute bit scaling factors for binary-to-continuous conversion
extern float BIT_SCALES[32]; // Supports up to 32 bits per dimension

#ifdef __cplusplus
}
#endif

#endif // KCA_H
