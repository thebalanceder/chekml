#ifndef RDA_H
#define RDA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define RDA_STEP_SIZE 0.1f
#define RDA_P_EXPLORATION 0.1f
#define RDA_POPULATION_SIZE 20
#define RDA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // RDA positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_float best_fitness;      // Best fitness value
} RDAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} RDACLContext;

// Function prototypes
void RDA_init_cl(RDACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void RDA_cleanup_cl(RDACLContext *cl_ctx);
void RDA_init_context(RDAContext *ctx, Optimizer *opt, RDACLContext *cl_ctx);
void RDA_cleanup_context(RDAContext *ctx, RDACLContext *cl_ctx);
void RDA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
