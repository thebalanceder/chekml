#ifndef SSA_H
#define SSA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SSA_MAX_GLIDING_DISTANCE 1.11f
#define SSA_MIN_GLIDING_DISTANCE 0.5f
#define SSA_GLIDING_CONSTANT 1.9f
#define SSA_NUM_FOOD_SOURCES 4
#define SSA_HICKORY_NUT_TREE 1
#define SSA_ACORN_NUT_TREE 3
#define SSA_NO_FOOD_TREES 46
#define SSA_POPULATION_SIZE 20
#define SSA_MAX_ITER 100

// Default maximum evaluations
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Squirrel positions
    cl_mem velocities;          // Squirrel velocities
    cl_mem fitness;             // Current fitness values
    cl_mem tree_types;          // Tree types (1: hickory, 2: normal, 3: acorn)
    cl_mem pulse_flying_rates;  // Pulse flying rates
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each squirrel
    cl_float best_fitness;      // Best fitness value
} SSAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating squirrel positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} SSACLContext;

// Function prototypes
void SSA_init_cl(SSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SSA_cleanup_cl(SSACLContext *cl_ctx);
void SSA_init_context(SSAContext *ctx, Optimizer *opt, SSACLContext *cl_ctx);
void SSA_cleanup_context(SSAContext *ctx, SSACLContext *cl_ctx);
void SSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
