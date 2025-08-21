#ifndef SFL_H
#define SFL_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SFL_MEMEPLEX_SIZE 10
#define SFL_NUM_MEMEPLEXES 5
#define SFL_NUM_PARENTS_RATIO 0.3f
#define SFL_NUM_OFFSPRINGS 3
#define SFL_MAX_FLA_ITER 5
#define SFL_STEP_SIZE 2.0f
#define SFL_POPULATION_SIZE (SFL_MEMEPLEX_SIZE * SFL_NUM_MEMEPLEXES)
#define SFL_MAX_ITER 500
#define SFL_MAX_EVALS_DEFAULT 10000

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_float best_fitness;      // Best fitness value
} SFLContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_memeplex_kernel; // Kernel for memeplex updates
    cl_bool owns_queue;         // Tracks if queue was created locally
} SFLCLContext;

// Function prototypes
void SFL_init_cl(SFLCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SFL_cleanup_cl(SFLCLContext *cl_ctx);
void SFL_init_context(SFLContext *ctx, Optimizer *opt, SFLCLContext *cl_ctx);
void SFL_cleanup_context(SFLContext *ctx, SFLCLContext *cl_ctx);
void SFL_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
