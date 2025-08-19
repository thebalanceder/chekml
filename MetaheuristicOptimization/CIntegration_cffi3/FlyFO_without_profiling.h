#ifndef FLYFO_H
#define FLYFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization Parameters
#define DELTASO_0 0.2
#define DELTASO_1 0.4
#define DELTASO_2 0.6
#define ALPHA_PARAM_0 1.0
#define ALPHA_PARAM_1 1.5
#define ALPHA_PARAM_2 1.9
#define PA_PARAM_0 0.5
#define PA_PARAM_1 0.85
#define PA_PARAM_2 0.99
#define SURVIVAL_LIST_RATIO 0.25
#define DEATH_PROB_FACTOR 1.0

// Algorithm Constants
#define MAX_EVALS_DEFAULT 100
#define POPULATION_SIZE_FACTOR 10
#define SQRT_DIM_FACTOR 2

// FFO Context for GPU
typedef struct {
    cl_mem population;          // GPU buffer: [pop_size * dim] positions
    cl_mem fitness;             // GPU buffer: [pop_size] fitness values
    cl_mem past_fitness;        // GPU buffer: [pop_size] past fitness
    cl_mem survival_list;       // GPU buffer: [surv_list_size * dim] positions
    cl_mem survival_fitness;    // GPU buffer: [surv_list_size] fitness
    cl_mem best_position;       // GPU buffer: [dim] best solution position
    cl_float best_fitness;      // Host: best fitness value
    cl_int survival_count;      // Host: number of survival list entries
    cl_float worst_fitness;     // Host: worst fitness in population
    cl_mem bounds;              // GPU buffer: [dim * 2] lower/upper bounds
    cl_mem random_seeds;        // GPU buffer: [pop_size] seeds for RNG
} FlyFOContext;

// OpenCL Context
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel fuzzy_tuning_kernel;
    cl_kernel update_position_kernel;
    cl_kernel crossover_kernel;
    cl_kernel suffocation_kernel;
    cl_device_id device;
} FlyFOCLContext;

// Function Declarations
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *));
void FlyFO_init_cl(FlyFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void FlyFO_cleanup_cl(FlyFOCLContext *cl_ctx);
void FlyFO_init_context(FlyFOContext *ctx, Optimizer *opt, FlyFOCLContext *cl_ctx);
void FlyFO_cleanup_context(FlyFOContext *ctx, FlyFOCLContext *cl_ctx);

#ifdef __cplusplus
}
#endif

#endif // FLYFO_H
