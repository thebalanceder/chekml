#ifndef BBO_H
#define BBO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define BBO_ALPHA 0.9f
#define MUTATION_PROB 0.1f
#define MUTATION_SCALE 0.02f
#define KEEP_RATE 0.2f
#define BBO_POPULATION_SIZE 20
#define BBO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Habitat positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each habitat
    cl_mem mu;                  // Emigration rates
    cl_mem lambda_;             // Immigration rates
    cl_float best_fitness;      // Best fitness value
} BBOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel migration_kernel; // Kernel for migration phase
    cl_kernel mutation_kernel;  // Kernel for mutation phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} BBOCLContext;

// Function prototypes
void BBO_init_cl(BBOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void BBO_cleanup_cl(BBOCLContext *cl_ctx);
void BBO_init_context(BBOContext *ctx, Optimizer *opt, BBOCLContext *cl_ctx);
void BBO_cleanup_context(BBOContext *ctx, BBOCLContext *cl_ctx);
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
