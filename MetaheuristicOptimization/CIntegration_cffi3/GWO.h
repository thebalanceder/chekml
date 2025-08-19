#ifndef GWO_H
#define GWO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define GWO_A_MAX 2.0f  // Initial value of coefficient 'a'
#define GWO_A_MIN 0.0f  // Final value of coefficient 'a'
#define GWO_POPULATION_SIZE 20
#define GWO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Wolf positions
    cl_mem fitness;             // Current fitness values
    cl_mem alpha_position;      // Alpha wolf position
    cl_mem beta_position;       // Beta wolf position
    cl_mem delta_position;      // Delta wolf position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each wolf
    cl_float alpha_fitness;     // Alpha fitness value
} GWOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating wolf positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} GWOCLContext;

// Function prototypes
void GWO_init_cl(GWOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void GWO_cleanup_cl(GWOCLContext *cl_ctx);
void GWO_init_context(GWOContext *ctx, Optimizer *opt, GWOCLContext *cl_ctx);
void GWO_cleanup_context(GWOContext *ctx, GWOCLContext *cl_ctx);
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
