#ifndef HPO_H
#define HPO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CONSTRICTION_COEFF 0.1
#define C_PARAM_MAX 0.98
#define TWO_PI 6.283185307179586

// Assume rand_double is defined elsewhere (e.g., WPA.h or generaloptimizer.c)
double rand_double(double min, double max);

// Helper for quicksort
void quicksort_indices(double *arr, int *indices, int low, int high);

// HPO Algorithm Phases
void hpo_update_positions(Optimizer *opt, int iter, double c_factor, double *xi, double *dist, int *idxsortdist, double *r1, double *r3, char *idx, double *z);

// Optimization Execution
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HPO_H
