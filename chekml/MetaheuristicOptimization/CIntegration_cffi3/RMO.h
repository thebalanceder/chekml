#ifndef RMO_H
#define RMO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define RMO_POPULATION_SIZE 50
#define RMO_MAX_ITERATIONS 100
#define RMO_ALPHA 0.1f

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Fitness values
    cl_mem bounds;              // Problem bounds
    cl_mem reference_point;     // Reference point
    cl_mem best_position;       // Best position found
    cl_mem random_seeds;        // Random seeds
    cl_float best_fitness;      // Best fitness value
} RMOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel sort_pop_kernel;  // Kernel for sorting population
    cl_kernel update_ref_kernel;// Kernel for updating reference point and positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} RMOCLContext;

// Function prototypes
void RMO_init_cl(RMOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void RMO_cleanup_cl(RMOCLContext *cl_ctx);
void RMO_init_context(RMOContext *ctx, Optimizer *opt, RMOCLContext *cl_ctx);
void RMO_cleanup_context(RMOContext *ctx, RMOCLContext *cl_ctx);
void RMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
