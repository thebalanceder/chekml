#ifndef CSS_H
#define CSS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include "generaloptimizer.h"

// Optimization parameters
#define KA 1.0
#define KV 1.0
#define A 1.0
#define EPSILON 1e-10
#define CM_SIZE_RATIO 0.25

// Utility function
double rand_double_css(double min, double max);

// CSS Algorithm Phases
void initialize_charged_particles(Optimizer *opt);
void calculate_forces(Optimizer *opt, double *forces);
void css_update_positions(Optimizer *opt, double *forces);
void update_charged_memory(Optimizer *opt, double *forces);

// Main Optimization Function
void CSS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CSS_H
