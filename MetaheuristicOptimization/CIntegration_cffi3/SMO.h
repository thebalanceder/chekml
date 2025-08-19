#ifndef SMO_H
#define SMO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SMO_PERTURBATION_RATE 0.1f
#define SMO_LOCAL_LEADER_LIMIT 50
#define SMO_GLOBAL_LEADER_LIMIT 1500
#define SMO_MAX_GROUPS 5
#define SMO_BHC_DELTA 0.1f
#define SMO_POPULATION_SIZE 20
#define SMO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Spider monkey positions
    cl_mem fitness;             // Current fitness values
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each spider monkey
    cl_mem group_ids;           // Group ID for each spider monkey
    cl_mem group_leaders;       // Local leader positions
    cl_mem group_leader_fitness; // Local leader fitness values
    cl_mem global_leader;       // Global leader position
    cl_float global_leader_fitness; // Global leader fitness
} SMOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel local_leader_kernel; // Kernel for local leader phase
    cl_kernel global_leader_kernel; // Kernel for global leader phase
    cl_kernel bhc_kernel;       // Kernel for beta-hill climbing
    cl_bool owns_queue;         // Tracks if queue was created locally
} SMOCLContext;

// Function prototypes
void SMO_init_cl(SMOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SMO_cleanup_cl(SMOCLContext *cl_ctx);
void SMO_init_context(SMOContext *ctx, Optimizer *opt, SMOCLContext *cl_ctx);
void SMO_cleanup_context(SMOContext *ctx, SMOCLContext *cl_ctx);
void SMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
