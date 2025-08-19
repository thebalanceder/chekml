#ifndef LSA_H
#define LSA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MAX_CHANNEL_TIME 100
#define LSA_ENERGY_FACTOR 2.05f
#define FOCKING_PROB 0.002f
#define DIRECTION_STEP 0.0f
#define LSA_POPULATION_SIZE 20
#define LSA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

// LSA context for GPU
typedef struct {
    cl_mem position;        // Channel positions
    cl_mem directions;      // Direction vectors
    cl_mem fitness;         // Fitness values
    cl_mem bounds;          // Problem bounds
    cl_mem random_seeds;    // Random seeds for each channel
    cl_float best_fitness; // Best fitness value
    cl_mem best_position;   // Best position
    cl_int channel_time;   // Channel time counter
} LSAContext;

typedef struct {
    cl_context context;          // OpenCL context
    cl_device_id device;         // OpenCL device
    cl_command_queue queue;      // OpenCL command queue
    cl_program program;          // OpenCL program
    cl_kernel init_kernel;       // Kernel for channel initialization
    cl_kernel update_kernel;     // Kernel for position updates
    cl_bool owns_queue;          // Tracks if queue was created locally
} LSACLContext;

// Function prototypes
void LSA_init_cl(LSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void LSA_cleanup_cl(LSACLContext *cl_ctx);
void LSA_init_context(LSAContext *ctx, Optimizer *opt, LSACLContext *cl_ctx);
void LSA_cleanup_context(LSAContext *ctx, LSACLContext *cl_ctx);
void LSA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#endif
