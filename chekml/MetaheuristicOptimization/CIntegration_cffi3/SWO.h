#ifndef SWO_H
#define SWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SWO_TRADE_OFF 0.3f
#define SWO_CROSSOVER_PROB 0.2f
#define SWO_MIN_POPULATION 20
#define SWO_LEVY_BETA 1.5f
#define SWO_LEVY_SCALE 0.05f
#define SWO_POPULATION_SIZE 20
#define SWO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_float best_fitness;      // Best fitness value
} SWOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel hunting_kernel;   // Kernel for hunting phase
    cl_kernel mating_kernel;    // Kernel for mating phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} SWOCLContext;

// Function prototypes
void SWO_init_cl(SWOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SWO_cleanup_cl(SWOCLContext *cl_ctx);
void SWO_init_context(SWOContext *ctx, Optimizer *opt, SWOCLContext *cl_ctx);
void SWO_cleanup_context(SWOContext *ctx, SWOCLContext *cl_ctx);
void SWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SWO_H
