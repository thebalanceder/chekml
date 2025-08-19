#ifndef BA_H
#define BA_H

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
#define LOUDNESS 1.0f       // Initial loudness (A)
#define PULSE_RATE 1.0f     // Initial pulse rate (r0)
#define ALPHA_BA 0.97f      // Loudness decay factor
#define GAMMA 0.1f          // Pulse rate increase factor
#define FREQ_MIN 0.0f       // Minimum frequency
#define FREQ_MAX 2.0f       // Maximum frequency
#define LOCAL_SEARCH_SCALE 0.1f // Scale for local search step
#define VELOCITY_MAX 10.0f  // Maximum velocity to prevent overshooting
#define POSITION_MAX 1000.0f // Safety bound for positions

// 🚀 Optimization Execution
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BA_H
