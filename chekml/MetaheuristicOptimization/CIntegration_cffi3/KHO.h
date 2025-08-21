#ifndef KHO_H
#define KHO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define KHO_VF 0.02f
#define KHO_DMAX 0.005f
#define KHO_NMAX 0.01f
#define KHO_CROSSOVER_RATE 0.8f
#define KHO_CROSSOVER_SCALE 0.2f
#define KHO_INERTIA_MIN 0.1f
#define KHO_INERTIA_MAX 0.8f
#define KHO_NEIGHBOR_LIMIT 4
#define KHO_SENSE_DISTANCE_FACTOR 5.0f
#define KHO_POPULATION_SIZE 20
#define KHO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Krill positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Global best position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each krill
    cl_mem N;                   // Induced motion vectors
    cl_mem F;                   // Foraging motion vectors
    cl_mem D;                   // Physical diffusion vectors
    cl_mem local_best_pos;      // Local best positions per krill
    cl_mem local_best_fitness;  // Local best fitness per krill
    cl_mem Xf;                  // Food position
    cl_mem K;                   // Current fitness buffer
    cl_mem Kib;                 // Local best fitness buffer
    cl_float best_fitness;      // Global best fitness
    cl_float Kf;                // Food fitness
} KHOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel motion_kernel;    // Kernel for movement-induced phase
    cl_kernel foraging_kernel;  // Kernel for foraging phase
    cl_kernel diffusion_kernel; // Kernel for physical diffusion
    cl_kernel crossover_kernel; // Kernel for crossover phase
    cl_kernel update_pos_kernel;// Kernel for position updates
    cl_bool owns_queue;         // Tracks if queue was created locally
} KHOCLContext;

// Function prototypes
void KHO_init_cl(KHOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void KHO_cleanup_cl(KHOCLContext *cl_ctx);
void KHO_init_context(KHOContext *ctx, Optimizer *opt, KHOCLContext *cl_ctx);
void KHO_cleanup_context(KHOContext *ctx, KHOCLContext *cl_ctx);
void KHO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
