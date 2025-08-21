#ifndef LOSA_H
#define LOSA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define LOSA_POPULATION_SIZE 50
#define LOSA_MAX_ITERATIONS 100
#define LOSA_STEP_SIZE 0.1f

typedef struct {
    cl_mem population;          // Particle positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Global best position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each particle
    cl_float best_fitness;      // Best fitness value
} LoSAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating particle positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} LoSACLContext;

// Function prototypes
void LoSA_init_cl(LoSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void LoSA_cleanup_cl(LoSACLContext *cl_ctx);
void LoSA_init_context(LoSAContext *ctx, Optimizer *opt, LoSACLContext *cl_ctx);
void LoSA_cleanup_context(LoSAContext *ctx, LoSACLContext *cl_ctx);
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
