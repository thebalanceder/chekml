#ifndef PVS_H
#define PVS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define PVS_DISTRIBUTION_INDEX 20
#define PVS_X_GAMMA 0.1
#define PVS_MAX_EVALS_DEFAULT 100000
#define PVS_CONVERGENCE_TOL 1e-6
#define PVS_STAGNATION_THRESHOLD 50

// PVS OpenCL Context
typedef struct {
    cl_mem positions;        // Particle positions
    cl_mem costs;            // Fitness costs
    cl_mem center;           // Vortex center
    cl_mem bounds;           // Search space bounds (lower, upper per dimension)
    cl_mem random_seeds;     // Random seeds for GPU
    cl_mem temp_solution;    // Temporary solution for crossover
    cl_mem mutated_solution; // Temporary solution for mutation
    cl_mem probabilities;    // Roulette wheel probabilities
    float best_cost;         // Best fitness value
} PVSContext;

// PVS OpenCL Program Context
typedef struct {
    cl_context context;                  // OpenCL context
    cl_device_id device;                 // OpenCL device
    cl_command_queue queue;              // Command queue
    cl_program program;                  // OpenCL program
    cl_kernel init_vortex_kernel;        // Kernel for vortex initialization
    cl_kernel first_phase_kernel;        // Kernel for first phase
    cl_kernel crossover_mutation_kernel; // Kernel for crossover and mutation
    cl_bool owns_queue;                  // Flag for queue ownership
} PVSCLContext;

// Utility function
static inline double rand_double_pvs(double min, double max);

// PVS Algorithm Phases
void PVS_init_cl(PVSCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void PVS_cleanup_cl(PVSCLContext *cl_ctx);
void PVS_init_context(PVSContext *ctx, Optimizer *opt, PVSCLContext *cl_ctx);
void PVS_cleanup_context(PVSContext *ctx, PVSCLContext *cl_ctx);

// Main Optimization Function
void PVS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // PVS_H
