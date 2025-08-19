#ifndef MFO_H
#define MFO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MFO_B_CONSTANT 1.0f
#define MFO_A_INITIAL -1.0f
#define MFO_A_FINAL -2.0f
#define MFO_POPULATION_SIZE 20
#define MFO_MAX_ITER 500
#define TWO_PI 6.283185307179586f

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Moth positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_flames;         // Best flame positions
    cl_mem best_flame_fitness;  // Best flame fitness values
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each moth
    cl_mem sorted_indices;      // Indices for sorted population
    cl_float best_fitness;      // Best fitness value
} MFOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating moth positions
    cl_kernel sort_kernel;      // Kernel for sorting population
    cl_kernel update_flames_kernel; // Kernel for updating flames
    cl_bool owns_queue;         // Tracks if queue was created locally
} MFOCLContext;

// Function prototypes
void MFO_init_cl(MFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void MFO_cleanup_cl(MFOCLContext *cl_ctx);
void MFO_init_context(MFOContext *ctx, Optimizer *opt, MFOCLContext *cl_ctx);
void MFO_cleanup_context(MFOContext *ctx, MFOCLContext *cl_ctx);
void MFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
