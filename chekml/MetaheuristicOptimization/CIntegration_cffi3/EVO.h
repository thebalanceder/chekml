#ifndef EVO_H
#define EVO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define EVO_STEP_SIZE 0.1f
#define EVO_MOMENTUM 0.9f
#define EVO_LEARNING_RATE 0.2f
#define EVO_POPULATION_SIZE 20
#define EVO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

// EVO context for GPU
typedef struct {
    cl_mem position;        // Particle positions
    cl_mem velocity;        // Particle velocities
    cl_mem gradient;        // Particle gradients
    cl_mem fitness;         // Fitness values
    cl_mem bounds;           // Problem bounds
    cl_mem random_seeds;    // Random seeds for each particle
    cl_float best_fitness; // Best fitness value
    cl_mem best_position;   // Best position
} EVOContext;

typedef struct {
    cl_context context;          // OpenCL context
    cl_device_id device;         // OpenCL device
    cl_command_queue queue;      // OpenCL command queue
    cl_program program;          // OpenCL program
    cl_kernel init_kernel;       // Kernel for population initialization
    cl_kernel gradient_kernel; // Kernel for gradient computation
    cl_kernel update_kernel; // Kernel for velocity/position updates
    cl_bool owns_queue;        // Tracks if queue was created locally
} EVOCLContext;

// Function prototypes
void EVO_init_cl(EVOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void EVO_cleanup_cl(EVOCLContext *cl_ctx);
void EVO_init_context(EVOContext *ctx, Optimizer *opt, EVOCLContext *cl_ctx);
void EVO_cleanup_context(EVOContext *ctx, EVOCLContext *cl_ctx);
void EVO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
