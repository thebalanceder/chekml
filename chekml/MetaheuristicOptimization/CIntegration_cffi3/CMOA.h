#ifndef CMOA_H
#define CMOA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CMOA_CROSSOVER_RATE 0.5f
#define CMOA_MUTATION_RATE 0.3f
#define CMOA_MUTATION_SCALE 0.1f
#define CMOA_POPULATION_SIZE 20
#define CMOA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Solution positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each solution
    cl_float best_fitness;      // Best fitness value
} CMOAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating solutions
    cl_bool owns_queue;         // Tracks if queue was created locally
} CMOACLContext;

// Function prototypes
void CMOA_init_cl(CMOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void CMOA_cleanup_cl(CMOACLContext *cl_ctx);
void CMOA_init_context(CMOAContext *ctx, Optimizer *opt, CMOACLContext *cl_ctx);
void CMOA_cleanup_context(CMOAContext *ctx, CMOACLContext *cl_ctx);
void CMOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
