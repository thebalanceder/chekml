#ifndef HPO_H
#define HPO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define HPO_CONSTRICTION_COEFF 0.1f
#define HPO_C_PARAM_MAX 0.98f
#define HPO_TWO_PI 6.283185307179586f
#define HPO_POPULATION_SIZE 20
#define HPO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_mem xi;                  // Mean position
    cl_mem dist;                // Distances to mean
    cl_mem idxsortdist;         // Sorted distance indices
    cl_float best_fitness;      // Best fitness value
} HPOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} HPOCLContext;

// Function prototypes
void HPO_init_cl(HPOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void HPO_cleanup_cl(HPOCLContext *cl_ctx);
void HPO_init_context(HPOContext *ctx, Optimizer *opt, HPOCLContext *cl_ctx);
void HPO_cleanup_context(HPOContext *ctx, HPOCLContext *cl_ctx);
void HPO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
