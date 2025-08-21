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

// WCA Parameters
#define PERTURBATION_FACTOR 0.1  // Stream position update range
#define LSAR_MAX_ITER 2000       // Maximum LSAR iterations
#define USE_LSAR_ASP 1           // Enable LSAR-ASP by default

// Utility Functions
double rand_double(double min, double max);
void enforce_bound_constraints(Optimizer *opt);

// WCA Algorithm Phases
void initialize_streams(Optimizer *opt);
void evaluate_streams(Optimizer *opt, double (*objective_function)(double *));
double iop(double **C, double **D, int rows, int cols_C, int cols_D);
int* positive_region(double **C, double **D, int rows, int cols_C, int cols_D, int *size);
int* fast_red(double **C, double **D, int rows, int cols_C, int cols_D, int *size);
void aps_mechanism(double **C, double **D, int rows, int cols_C, int cols_D, int *B, int B_size, int *u, int *v);
int* lsar_asp(double **C, double **D, int rows, int cols_C, int cols_D, int *size);

// Main Optimization Function
void WCA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WCA_H
