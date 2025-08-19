#ifndef POA_H
#define POA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define POA_STEP_SIZE 0.1f
#define POA_POPULATION_SIZE 20
#define POA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Particle positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each particle
    cl_float best_fitness;      // Best fitness value
} POAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating particle positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} POACLContext;

// Function prototypes
void POA_init_cl(POACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void POA_cleanup_cl(POACLContext *cl_ctx);
void POA_init_context(POAContext *ctx, Optimizer *opt, POACLContext *cl_ctx);
void POA_cleanup_context(POAContext *ctx, POACLContext *cl_ctx);
void POA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
