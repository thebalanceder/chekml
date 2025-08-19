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

// Inline utility function
static inline double gso_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// GSO Algorithm Phases
void luciferin_update(Optimizer *opt, double (*objective_function)(double *));
void movement_phase(Optimizer *opt, double *decision_range, double *distances, int *neighbors, double *probs, double *current_pos);
void decision_range_update(Optimizer *opt, double *decision_range, double *distances, int *neighbors);

// Main Optimization Function
void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GLOWSO_H
