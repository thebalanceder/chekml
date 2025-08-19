#ifndef TFWO_H
#define TFWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define N_WHIRLPOOLS_DEFAULT 3
#define N_OBJECTS_PER_WHIRLPOOL_DEFAULT 30
#ifndef PI
#define PI 3.14159265358979323846f
#endif
#define TFWO_MAX_EVALS_DEFAULT 100000
#define TFWO_CONVERGENCE_TOL 1e-6
#define TFWO_STAGNATION_THRESHOLD 50

// TFWO OpenCL Context
typedef struct {
    cl_mem wp_positions;     // Whirlpool positions
    cl_mem wp_costs;         // Whirlpool costs
    cl_mem wp_deltas;        // Whirlpool angular displacements
    cl_mem wp_position_sums; // Whirlpool position sums
    cl_mem obj_positions;    // Object positions
    cl_mem obj_costs;        // Object costs
    cl_mem obj_deltas;       // Object angular displacements
    cl_mem bounds;           // Search space bounds
    cl_mem random_seeds;     // Random seeds for GPU
    cl_mem temp_d;           // Temporary array d
    cl_mem temp_d2;          // Temporary array d2
    cl_mem temp_RR;          // Temporary array RR
    cl_mem temp_J;           // Temporary array J
    float best_cost;         // Best fitness value
} TFWOContext;

// TFWO OpenCL Program Context
typedef struct {
    cl_context context;                  // OpenCL context
    cl_device_id device;                 // OpenCL device
    cl_command_queue queue;              // Command queue
    cl_program program;                  // OpenCL program
    cl_kernel init_whirlpools_kernel;    // Kernel for whirlpool initialization
    cl_kernel effects_whirlpools_kernel; // Kernel for computing whirlpool effects
    cl_kernel update_best_kernel;        // Kernel for updating best whirlpool
    cl_bool owns_queue;                  // Flag for queue ownership
} TFWOCLContext;

// Utility function
static inline double rand_double_tfwo(double min, double max);

// TFWO Algorithm Phases
void TFWO_init_cl(TFWOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void TFWO_cleanup_cl(TFWOCLContext *cl_ctx);
void TFWO_init_context(TFWOContext *ctx, Optimizer *opt, TFWOCLContext *cl_ctx);
void TFWO_cleanup_context(TFWOContext *ctx, TFWOCLContext *cl_ctx);

// Main Optimization Function
void TFWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TFWO_H
