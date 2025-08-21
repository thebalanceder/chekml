#ifndef BHA_H
#define BHA_H

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
#define BHA_POPULATION_SIZE 50
#define BHA_MAX_ITER 500
#define BHA_EPSILON 1e-10f

// BHA OpenCL Context
typedef struct {
    cl_mem positions;        // Star positions
    cl_mem fitness;          // Fitness values
    cl_mem bounds;           // Search space bounds
    cl_mem random_seeds;     // Random seeds for GPU
    cl_mem best_index;       // Index of the black hole
    float best_fitness;      // Best fitness value
} BHAContext;

// BHA OpenCL Program Context
typedef struct {
    cl_context context;                  // OpenCL context
    cl_device_id device;                 // OpenCL device
    cl_command_queue queue;              // Command queue
    cl_program program;                  // OpenCL program
    cl_kernel init_stars_kernel;         // Kernel for star initialization
    cl_kernel update_positions_kernel;   // Kernel for position updates
    cl_kernel new_star_gen_kernel;       // Kernel for new star generation
    cl_kernel find_black_hole_kernel;    // Kernel for finding black hole
    cl_bool owns_queue;                  // Flag for queue ownership
} BHACLContext;

// Utility function
static inline double rand_double_bha(double min, double max);

// BHA Algorithm Phases
void BHA_init_cl(BHACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void BHA_cleanup_cl(BHACLContext *cl_ctx);
void BHA_init_context(BHAContext *ctx, Optimizer *opt, BHACLContext *cl_ctx);
void BHA_cleanup_context(BHAContext *ctx, BHACLContext *cl_ctx);

// Main Optimization Function
void BHA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BHA_H
