#ifndef SEO_H
#define SEO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SEO_POPULATION_SIZE 50
#define SEO_MAX_ITER 100
#define SEO_RAND_SCALE 0.1f  // Scaling factor for random perturbations

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

// SEO context for OpenCL buffers
typedef struct {
    cl_mem population;          // Agent positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each agent
    cl_float best_fitness;      // Best fitness value
} SEOContext;

// SEO OpenCL context
typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for social engineering update
    cl_bool owns_queue;         // Tracks if queue was created locally
} SEOCLContext;

// Function prototypes
void SEO_init_cl(SEOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SEO_cleanup_cl(SEOCLContext *cl_ctx);
void SEO_init_context(SEOContext *ctx, Optimizer *opt, SEOCLContext *cl_ctx);
void SEO_cleanup_context(SEOContext *ctx, SEOCLContext *cl_ctx);
void SEO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SEO_H
