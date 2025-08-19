#ifndef DHLO_H
#define DHLO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define INITIAL_LEADERS 30
#define TOLERANCE_DEFAULT 5.0
#define A_MAX 1.0f
#define A_MIN 0.0f
#define DHLO_POP_DEFAULT 50
#define DHLO_MAX_ITER 500

#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

// DHLO configuration
#define DHLO_VARIANT "V4"  // Options: V1, V2, V3, V4
#define POSITION_UPDATE_STRATEGY "GWO"  // Options: GWO, PSO, RW

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Fitness values
    cl_mem leaders;            // Leaders positions
    cl_mem leader_fitness;     // Leaders fitness
    cl_mem pbest;              // Personal best positions
    cl_mem pbest_fitness;      // Personal best fitness
    cl_mem best_solution;      // Best solution position
    cl_mem bounds;             // Problem bounds
    cl_mem random_seeds;       // Random seeds
    cl_float best_fitness;     // Best fitness value
} DHLOContext;

typedef struct {
    cl_context context;        // OpenCL context
    cl_device_id dev;          // OpenCL device
    cl_command_queue cl_ctx;   // Command queue
    cl_program cl_program;     // OpenCL program
    cl_kernel init_kernel;     // Kernel for initialization
    cl_kernel update_kernel;   // Kernel for position updates
    cl_bool owns_queue;        // Tracks queue ownership
} DHLCLContext;

// Function prototypes
void DHLO_init_cl(DHLCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void DHLO_cleanup_cl(DHLCLContext *cl_ctx);
void DHLO_init_context(DHLOContext *ctx, Optimizer *opt, DHLCLContext *cl_ctx);
void DHLO_cleanup_context(DHLOContext *ctx, DHLCLContext *cl_ctx);
void DHLO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
