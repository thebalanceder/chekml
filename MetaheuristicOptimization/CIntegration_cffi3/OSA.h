#ifndef OSA_H
#define OSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define OSA_STEP_SIZE 0.1f
#define OSA_P_EXPLORE 0.1f
#define OSA_POPULATION_SIZE 20
#define OSA_MAX_ITER 500

// Default maximum evaluations
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds
    cl_float best_fitness;      // Best fitness value
} OSAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for position updates
    cl_bool owns_queue;         // Tracks queue ownership
} OSACLContext;

// Function prototypes
void OSA_init_cl(OSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void OSA_cleanup_cl(OSACLContext *cl_ctx);
void OSA_init_context(OSAContext *ctx, Optimizer *opt, OSACLContext *cl_ctx);
void OSA_cleanup_context(OSAContext *ctx, OSACLContext *cl_ctx);
void OSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // OSA_H
