#ifndef FSA_H
#define FSA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define FSA_MAX_ITERATIONS 100
#define FSA_NUM_RUNS 30
#define FSA_POPULATION_SIZE 50
#define FSA_DEBUG 0  // Set to 1 to enable logging

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem local_best_positions;// Local best positions for each solution
    cl_mem local_best_values;   // Local best fitness values
    cl_mem best_position;       // Global best position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each solution
    cl_float best_fitness;      // Global best fitness value
} FSAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_pop_kernel;// Kernel for population update
    cl_kernel initial_strategy_kernel; // Kernel for initial strategy update
    cl_bool owns_queue;         // Tracks if queue was created locally
} FSACLContext;

// Function prototypes
void FSA_init_cl(FSACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void FSA_cleanup_cl(FSACLContext *cl_ctx);
void FSA_init_context(FSAContext *ctx, Optimizer *opt, FSACLContext *cl_ctx);
void FSA_cleanup_context(FSAContext *ctx, FSACLContext *cl_ctx);
void FSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
