#ifndef IWD_H
#define IWD_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define IWD_A_S 1.0f
#define IWD_B_S 0.01f
#define IWD_C_S 1.0f
#define IWD_A_V 1.0f
#define IWD_B_V 0.01f
#define IWD_C_V 1.0f
#define IWD_INIT_VEL 200.0f
#define IWD_P_N 0.9f
#define IWD_P_IWD 0.9f
#define IWD_INITIAL_SOIL 10000.0f
#define IWD_EPSILON_S 0.0001f
#define IWD_POPULATION_SIZE 20
#define IWD_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Water drop positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each water drop
    cl_mem soil;                // Soil matrix
    cl_mem hud;                 // Heuristic undesirability matrix
    cl_mem velocities;          // Velocities of water drops
    cl_float best_fitness;      // Best fitness value
} IWDContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel move_drop_kernel; // Kernel for moving water drops
    cl_kernel update_soil_kernel; // Kernel for updating soil
    cl_bool owns_queue;         // Tracks if queue was created locally
} IWDCLContext;

// Function prototypes
void IWD_init_cl(IWDCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void IWD_cleanup_cl(IWDCLContext *cl_ctx);
void IWD_init_context(IWDContext *ctx, Optimizer *opt, IWDCLContext *cl_ctx);
void IWD_cleanup_context(IWDContext *ctx, IWDCLContext *cl_ctx);
void IWD_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
