#ifndef TLBO_H
#define TLBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// TLBO Context for GPU
typedef struct {
    cl_mem positions;       // Population positions
    cl_mem costs;           // Fitness values
    cl_mem best_position;   // Best solution position
    cl_mem bounds;          // Bounds for each dimension
    cl_mem random_seeds;    // Random seeds for GPU
    cl_mem partners;        // Partner indices for learner phase
    cl_mem mean_student;    // Mean of population per dimension
    double best_cost;       // Best fitness value
} TLBOContext;

// TLBO OpenCL Context
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_population_kernel;
    cl_kernel compute_mean_kernel;
    cl_kernel teacher_phase_kernel;
    cl_kernel learner_phase_kernel;
    cl_bool owns_queue;
} TLBOCLContext;

// Constants
#define TLBO_MAX_EVALS_DEFAULT 10000
#define TLBO_CONVERGENCE_TOL 1e-6
#define TLBO_STAGNATION_THRESHOLD 10

// Function Prototypes
void TLBO_init_cl(TLBOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void TLBO_cleanup_cl(TLBOCLContext *cl_ctx);
void TLBO_init_context(TLBOContext *ctx, Optimizer *opt, TLBOCLContext *cl_ctx);
void TLBO_cleanup_context(TLBOContext *ctx, TLBOCLContext *cl_ctx);
void TLBO_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // TLBO_H
