#ifndef SOA_H
#define SOA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SOA_MU_MAX 0.9f
#define SOA_MU_MIN 0.05f
#define W_MAX_SOA 0.9f
#define W_MIN_SOA 0.2f
#define NUM_REGIONS 3
#define SOA_MAX_EVALS_DEFAULT 100

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem pbest_s;            // Personal best positions
    cl_mem pbest_fun;          // Personal best fitness values
    cl_mem lbest_s;            // Local best positions
    cl_mem lbest_fun;          // Local best fitness values
    cl_mem e_t_1;              // Previous positions (t-1)
    cl_mem e_t_2;              // Previous positions (t-2)
    cl_mem f_t_1;              // Previous fitness (t-1)
    cl_mem f_t_2;              // Previous fitness (t-2)
    cl_mem bounds;             // Problem bounds
    cl_mem random_seeds;       // Random seeds for each particle
    cl_mem start_reg;          // Region start indices
    cl_mem end_reg;            // Region end indices
    cl_mem size_reg;           // Region sizes
    cl_mem rmax;               // Maximum step sizes
    cl_mem rmin;               // Minimum step sizes
    cl_float best_fitness;     // Best fitness value
    cl_mem best_position;      // Best position found
} SOAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_kernel;    // Kernel for updating positions
    cl_bool owns_queue;         // Tracks if queue was created locally
} SOACLContext;

// Function prototypes
void SOA_init_cl(SOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SOA_cleanup_cl(SOACLContext *cl_ctx);
void SOA_init_context(SOAContext *ctx, Optimizer *opt, SOACLContext *cl_ctx);
void SOA_cleanup_context(SOAContext *ctx, SOACLContext *cl_ctx);
void SOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
