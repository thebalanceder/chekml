#ifndef WO_H
#define WO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define WO_FEMALE_PROPORTION 0.4
#define WO_HALTON_BASE 7
#define WO_LEVY_BETA 1.5

// Helper functions
double WO_rand_double(double min, double max);
double WO_halton_sequence(int index, int base);
void WO_levy_flight(double *step, int dim);

// WO Algorithm Phases
void WO_migration_phase(Optimizer *opt, double beta, double r3);
void WO_male_position_update(Optimizer *opt);
void WO_female_position_update(Optimizer *opt, double alpha);
void WO_child_position_update(Optimizer *opt);
void WO_position_adjustment_phase(Optimizer *opt, double R);
void WO_exploitation_phase(Optimizer *opt, double beta);

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WO_H
