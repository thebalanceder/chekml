#ifndef BA_H
#define BA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define LOUDNESS 1.0        // Initial loudness (A)
#define PULSE_RATE 1.0      // Initial pulse rate (r0)
#define ALPHA_BA 0.97          // Loudness decay factor
#define GAMMA 0.1           // Pulse rate increase factor
#define FREQ_MIN 0.0        // Minimum frequency
#define FREQ_MAX 2.0        // Maximum frequency
#define LOCAL_SEARCH_SCALE 0.1  // Scale for local search step

// 🦇 Inline utility function
static inline double rand_double_ba(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🦇 Bat Algorithm Phases
void bat_frequency_update(Optimizer *restrict opt, double *restrict freq, double *restrict velocities);
void bat_local_search(Optimizer *restrict opt, double *restrict freq, double pulse_rate, double loudness);
void bat_update_solutions(Optimizer *restrict opt, double *restrict freq, double loudness, double (*objective_function)(double *));

// 🚀 Optimization Execution
void BA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BA_H
