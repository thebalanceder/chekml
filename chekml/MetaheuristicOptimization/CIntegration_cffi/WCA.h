#ifndef WCA_H
#define WCA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// WCA Parameters
#define PERTURBATION_FACTOR 0.1
#define LSAR_MAX_ITER 2000
#define USE_LSAR_ASP 1
#define MAX_DIM 100
#define EARLY_STOP_THRESHOLD 1e-6

// Utility Functions
void wca_enforce_bound_constraints(Optimizer *opt);

// WCA Algorithm Phases
void initialize_streams(Optimizer *opt);
void evaluate_streams(Optimizer *opt, double (*objective_function)(double *));
double iop(double *C, double *D, int rows, int cols_C, int cols_D, int *cache);
int* positive_region(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache);
int* fast_red(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache);
void aps_mechanism(double *C, double *D, int rows, int cols_C, int cols_D, int *B, int B_size, int *u, int *v, int *cache);
int* lsar_asp(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache);

// Main Optimization Function
void WCA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WCA_H
