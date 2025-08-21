#ifndef ICA_H
#define ICA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define ICA_NUM_IMPERIALISTS 8
#define ICA_NUM_COUNTRIES 100
#define ICA_ZETA 0.1f
#define ICA_ASSIMILATION_COEFF 2.0f
#define ICA_REVOLUTION_RATE 0.3f
#define ICA_DAMP_RATIO 0.99f
#define ICA_UNITING_THRESHOLD 0.02f
#define ICA_MAX_DECADES 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem positions;           // Country positions
    cl_mem costs;               // Country costs
    cl_mem empire_indices;      // Empire assignment for each country
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each country
    cl_mem best_position;       // Best position found
    cl_float best_cost;         // Best cost value
} ICAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_countries_kernel; // Kernel for country initialization
    cl_kernel assimilate_kernel;     // Kernel for assimilating colonies
    cl_bool owns_queue;         // Tracks if queue was created locally
} ICACLContext;

// Function prototypes
void ICA_init_cl(ICACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void ICA_cleanup_cl(ICACLContext *cl_ctx);
void ICA_init_context(ICAContext *ctx, Optimizer *opt, ICACLContext *cl_ctx);
void ICA_cleanup_context(ICAContext *ctx, ICACLContext *cl_ctx);
void ICA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
