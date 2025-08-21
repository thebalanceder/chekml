#ifndef SASA_H
#define SASA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SASA_POPULATION_SIZE 50
#define SASA_MAX_ITERATIONS 100
#define SASA_C1_FACTOR 2.0f
#define SASA_C1_EXPONENT 4.0f

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Salp positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each salp
    cl_float best_fitness;      // Best fitness value
} SaSAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating salp positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} SaSACLContext;

// Function prototypes
void SaSA_init_cl(SaSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SaSA_cleanup_cl(SaSACLContext *cl_ctx);
void SaSA_init_context(SaSAContext *ctx, Optimizer *opt, SaSACLContext *cl_ctx);
void SaSA_cleanup_context(SaSAContext *ctx, SaSACLContext *cl_ctx);
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SASA_H
