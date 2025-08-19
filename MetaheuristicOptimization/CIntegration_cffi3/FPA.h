#ifndef FPA_H
#define FPA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define FPA_SWITCH_PROB 0.8f
#define FPA_LEVY_STEP_SCALE 0.01f
#define FPA_LEVY_BETA 1.5f
#define FPA_LEVY_SIGMA 0.6966f  // Precomputed for beta = 1.5
#define FPA_INV_LEVY_BETA (1.0f / 1.5f)  // Precomputed 1/beta
#define FPA_MAX_EVALS_DEFAULT 100000

// FPA Constants
#define FPA_POPULATION_SIZE_DEFAULT 20
#define FPA_MAX_ITER_DEFAULT 5000

// OpenCL context structure
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel global_poll_kernel;
    cl_kernel local_poll_kernel;
    cl_bool owns_queue;
} FPA_CLContext;

// FPA Context structure
typedef struct {
    cl_mem population;
    cl_mem fitness;
    cl_mem bounds;
    cl_mem best_position;
    cl_mem random_seeds;
    float best_fitness;
} FPA_Context;

// FPA Algorithm Phases
void fpa_init_cl(FPA_CLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void fpa_cleanup_cl(FPA_CLContext *cl_ctx);
void fpa_init_context(FPA_Context *ctx, Optimizer *opt, FPA_CLContext *cl_ctx);
void fpa_cleanup_context(FPA_Context *ctx, FPA_CLContext *cl_ctx);

// Optimization Execution
void FPA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FPA_H
