#ifndef DRA_H
#define DRA_H

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
#define DRA_BELIEF_PROFILE_RATE 0.5f
#define DRA_MIRACLE_RATE 0.5f
#define DRA_PROSELYTISM_RATE 0.9f
#define DRA_REWARD_PENALTY_RATE 0.2f
#define DRA_NUM_GROUPS 5
#define DRA_MAX_EVALS_DEFAULT 10000
#define DRA_CONVERGENCE_TOL 1e-6
#define DRA_STAGNATION_THRESHOLD 10
#ifndef PI
#define PI 3.141592653589793f
#endif

// DRA Context for GPU
typedef struct {
    cl_mem positions;       // Population positions
    cl_mem costs;           // Fitness values
    cl_mem best_position;   // Best solution position
    cl_mem bounds;          // Bounds for each dimension
    cl_mem random_seeds;    // Random seeds for GPU
    cl_mem min_max_indices; // Indices for min/max fitness
    double best_cost;       // Best fitness value
} DRAContext;

// DRA OpenCL Context
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_belief_profiles_kernel;
    cl_kernel miracle_operator_kernel;
    cl_kernel proselytism_operator_kernel;
    cl_kernel reward_penalty_operator_kernel;
    cl_kernel replacement_operator_kernel;
    cl_kernel find_min_max_kernel;
    cl_bool owns_queue;
} DRACLContext;

// Function Prototypes
void DRA_init_cl(DRACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void DRA_cleanup_cl(DRACLContext *cl_ctx);
void DRA_init_context(DRAContext *ctx, Optimizer *opt, DRACLContext *cl_ctx);
void DRA_cleanup_context(DRAContext *ctx, DRACLContext *cl_ctx);
void DRA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // DRA_H
