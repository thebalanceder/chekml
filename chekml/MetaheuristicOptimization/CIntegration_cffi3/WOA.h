// File: WOA.h
#ifndef WOA_H
#define WOA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define WOA_A_INITIAL 2.0f
#define WOA_A2_INITIAL -1.0f
#define WOA_A2_FINAL -2.0f
#define WOA_B 1.0f
#define WOA_POPULATION_SIZE 20
#define WOA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Whale positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each whale
    cl_float best_fitness;      // Best fitness value
} WOAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating whale positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} WOACLContext;

// Function prototypes
void WOA_init_cl(WOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void WOA_cleanup_cl(WOACLContext *cl_ctx);
void WOA_init_context(WOAContext *ctx, Optimizer *opt, WOACLContext *cl_ctx);
void WOA_cleanup_context(WOAContext *ctx, WOACLContext *cl_ctx);
void WOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
