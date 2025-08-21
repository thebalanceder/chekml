#ifndef WO_H
#define WO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define WO_FEMALE_PROPORTION 0.4f
#define WO_HALTON_BASE 7
#define WO_LEVY_BETA 1.5f
#define WO_POPULATION_SIZE 20
#define WO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

// Context struct for WO-specific state
typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem second_best;         // Second-best position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_mem temp_indices;        // Temporary indices for migration
    cl_mem temp_array1;         // Temporary array for levy flight
    cl_mem temp_array2;         // Temporary array for child updates
    cl_float best_fitness;      // Best fitness value
    cl_float second_best_fitness; // Second-best fitness value
    cl_int male_count;          // Number of male individuals
    cl_int female_count;        // Number of female individuals
    cl_int child_count;         // Number of child individuals
    cl_float levy_sigma;        // Levy flight sigma
} WOContext;

// OpenCL context struct
typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel migration_kernel; // Kernel for migration phase
    cl_kernel male_update_kernel; // Kernel for male position update
    cl_kernel female_update_kernel; // Kernel for female position update
    cl_kernel child_update_kernel; // Kernel for child position update
    cl_kernel pos_adjust_kernel; // Kernel for position adjustment
    cl_kernel exploitation_kernel; // Kernel for exploitation phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} WOCLContext;

// Function prototypes
void WO_init_cl(WOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void WO_cleanup_cl(WOCLContext *cl_ctx);
void WO_init_context(WOContext *ctx, Optimizer *opt, WOCLContext *cl_ctx);
void WO_cleanup_context(WOContext *ctx, WOCLContext *cl_ctx);
void WO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WO_H
