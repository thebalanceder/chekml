#ifndef AMO_H
#define AMO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define AMO_NEIGHBORHOOD_SIZE 5
#define AMO_POPULATION_SCALE_FACTOR 1.0f
#define AMO_MIGRATION_PROBABILITY_FACTOR 0.5f
#define AMO_NEIGHBORHOOD_RADIUS 2
#define AMO_FITNESS_SCALING 1.0f
#define AMO_POPULATION_SIZE 20
#define AMO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_float best_fitness;      // Best fitness value
} AMOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel neighbor_learn_kernel; // Kernel for neighborhood learning phase
    cl_kernel global_migrate_kernel; // Kernel for global migration phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} AMOCLContext;

// Function prototypes
void AMO_init_cl(AMOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void AMO_cleanup_cl(AMOCLContext *cl_ctx);
void AMO_init_context(AMOContext *ctx, Optimizer *opt, AMOCLContext *cl_ctx);
void AMO_cleanup_context(AMOContext *ctx, AMOCLContext *cl_ctx);
void AMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
