#ifndef HHO_H
#define HHO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// HHO Parameters
#define HHO_BETA 1.5f  // Levy flight parameter
#define ENERGY_FACTOR 2.0f  // Initial energy factor
#define PI 3.14159265358979323846f

typedef struct {
    cl_mem population;        // Population positions
    cl_mem fitness;           // Current fitness values
    cl_mem best_position;     // Best solution position
    cl_mem bounds;            // Problem bounds
    cl_mem random_seeds;      // Seeds for RNG
    cl_float best_fitness;    // Best fitness value
} HHOContext;

typedef struct {
    cl_context context;       // OpenCL context
    cl_device_id device;      // OpenCL device
    cl_command_queue queue;   // OpenCL command queue
    cl_program program;       // OpenCL program
    cl_kernel init_pop_kernel;       // Kernel for population initialization
    cl_kernel exploration_kernel;     // Kernel for exploration phase
    cl_kernel exploitation_kernel;    // Kernel for exploitation phase
    cl_kernel levy_flight_kernel;     // Kernel for Levy flight
    cl_bool owns_queue;       // Tracks if queue was created locally
} HHOCLContext;

void HHO_init_cl(HHOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void HHO_cleanup_cl(HHOCLContext *cl_ctx);
void HHO_init_context(HHOContext *ctx, Optimizer *opt, HHOCLContext *cl_ctx);
void HHO_cleanup_context(HHOContext *ctx, HHOCLContext *cl_ctx);
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
