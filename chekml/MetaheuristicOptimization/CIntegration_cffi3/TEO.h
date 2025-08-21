#ifndef TEO_H
#define TEO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define TEO_STEP_SIZE 0.1f
#define TEO_INITIAL_TEMPERATURE 100.0f
#define TEO_FINAL_TEMPERATURE 0.01f
#define TEO_COOLING_RATE 0.99f
#define TEO_POPULATION_SIZE 20
#define TEO_MAX_ITER 500

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
} TEOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel perturb_kernel;   // Kernel for perturbing solutions
    cl_bool owns_queue;         // Tracks if queue was created locally
} TEOCLContext;

// Function prototypes
void TEO_init_cl(TEOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void TEO_cleanup_cl(TEOCLContext *cl_ctx);
void TEO_init_context(TEOContext *ctx, Optimizer *opt, TEOCLContext *cl_ctx);
void TEO_cleanup_context(TEOContext *ctx, TEOCLContext *cl_ctx);
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
