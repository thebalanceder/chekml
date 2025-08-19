#ifndef GLOWSO_H
#define GLOWSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define LUCIFERIN_INITIAL 5.0
#define DECISION_RANGE_INITIAL 3.0
#define LUCIFERIN_DECAY 0.4
#define LUCIFERIN_ENHANCEMENT 0.6
#define NEIGHBOR_THRESHOLD 0.08
#define GSO_STEP_SIZE 0.6
#define SENSOR_RANGE 10.0
#define NEIGHBOR_COUNT 10

// Inline utility function for random double
static inline double gso_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// GSO Algorithm Phases
void luciferin_update(Optimizer *restrict opt, double (*objective_function)(double *));
void movement_phase(Optimizer *restrict opt, double *restrict decision_range, double *restrict distances, int *restrict neighbors, double *restrict probs, double *restrict current_pos);
void decision_range_update(Optimizer *restrict opt, double *restrict decision_range, double *restrict distances, int *restrict neighbors);

// Main Optimization Function
void GlowSO_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GLOWSO_H
