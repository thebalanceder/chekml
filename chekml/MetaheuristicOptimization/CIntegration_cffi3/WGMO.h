#ifndef WGMO_H
#define WGMO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define WGMO_ALPHA 0.9f
#define WGMO_BETA 0.1f
#define WGMO_GAMMA 0.1f
#define WGMO_POPULATION_SIZE 20
#define WGMO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Geese positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each goose
    cl_float best_fitness;      // Best fitness value
} WGMOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating geese positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} WGMOCLContext;

// Function prototypes
void WGMO_init_cl(WGMOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void WGMO_cleanup_cl(WGMOCLContext *cl_ctx);
void WGMO_init_context(WGMOContext *ctx, Optimizer *opt, WGMOCLContext *cl_ctx);
void WGMO_cleanup_context(WGMOContext *ctx, WGMOCLContext *cl_ctx);
void WGMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
