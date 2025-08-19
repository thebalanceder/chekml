#ifndef FIREFA_H
#define FIREFA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define FA_ALPHA 1.0f
#define FA_BETA0 1.0f
#define FA_GAMMA 0.01f
#define FA_THETA 0.97f
#define FA_POPULATION_SIZE 20
#define FA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Firefly positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each firefly
    cl_float best_fitness;      // Best fitness value
} FirefAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating firefly positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} FirefACLContext;

// Function prototypes
void FirefA_init_cl(FirefACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void FirefA_cleanup_cl(FirefACLContext *cl_ctx);
void FirefA_init_context(FirefAContext *ctx, Optimizer *opt, FirefACLContext *cl_ctx);
void FirefA_cleanup_context(FirefAContext *ctx, FirefACLContext *cl_ctx);
void FirefA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
