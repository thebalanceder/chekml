#ifndef CSS_H
#define CSS_H

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
#define CSS_KA 1.0
#define CSS_KV 1.0
#define CSS_A 1.0
#define CSS_EPSILON 1e-10
#define CSS_CM_SIZE_RATIO 0.25
#define CSS_MAX_EVALS_DEFAULT 100000
#define CSS_CONVERGENCE_TOL 1e-6
#define CSS_STAGNATION_THRESHOLD 50

// CSS OpenCL Context
typedef struct {
    cl_mem positions;        // Particle positions
    cl_mem costs;            // Fitness costs
    cl_mem forces;           // Resultant forces
    cl_mem velocities;       // Particle velocities
    cl_mem bounds;           // Search space bounds
    cl_mem random_seeds;     // Random seeds for GPU
    cl_mem cm_positions;     // Charged memory positions
    cl_mem cm_costs;         // Charged memory costs
    cl_mem min_max_indices;  // Indices of min/max costs
    float best_cost;         // Best fitness value
} CSSContext;

// CSS OpenCL Program Context
typedef struct {
    cl_context context;                  // OpenCL context
    cl_device_id device;                 // OpenCL device
    cl_command_queue queue;              // Command queue
    cl_program program;                  // OpenCL program
    cl_kernel init_particles_kernel;     // Kernel for particle initialization
    cl_kernel calc_forces_kernel;        // Kernel for force calculation
    cl_kernel update_positions_kernel;   // Kernel for position updates
    cl_kernel update_cm_kernel;          // Kernel for charged memory update
    cl_kernel find_min_max_kernel;       // Kernel for finding min/max indices
    cl_bool owns_queue;                  // Flag for queue ownership
} CSSCLContext;

// Utility function
static inline double rand_double_css(double min, double max);

// CSS Algorithm Phases
void CSS_init_cl(CSSCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void CSS_cleanup_cl(CSSCLContext *cl_ctx);
void CSS_init_context(CSSContext *ctx, Optimizer *opt, CSSCLContext *cl_ctx);
void CSS_cleanup_context(CSSContext *ctx, CSSCLContext *cl_ctx);

// Main Optimization Function
void CSS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CSS_H
