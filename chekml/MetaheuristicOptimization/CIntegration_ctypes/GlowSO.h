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

// GSO Algorithm Phases
void luciferin_update(Optimizer *opt, double (*objective_function)(double *));
void movement_phase_glowso(Optimizer *opt, double *decision_range);
void decision_range_update(Optimizer *opt, double *decision_range);

// Main Optimization Function
void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GLOWSO_H
