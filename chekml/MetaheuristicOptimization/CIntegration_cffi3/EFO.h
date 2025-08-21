#ifndef EFO_H
#define EFO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define EFO_RANDOMIZATION_RATE 0.3f
#define EFO_POSITIVE_SELECTION_RATE 0.2f
#define EFO_POSITIVE_FIELD_RATIO 0.1f
#define EFO_NEGATIVE_FIELD_RATIO 0.45f
#define EFO_GOLDEN_RATIO 1.618033988749895f
#define EFO_POPULATION_SIZE 20
#define EFO_MAX_ITER 100

// Default maximum evaluations if not specified
#define MAX_EVALS_DEFAULT 100

// EFO context for GPU
typedef struct {
    cl_mem position;            // Particle positions
    cl_mem fitness;             // Fitness values
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each particle
    cl_float best_fitness;      // Best fitness value
    cl_mem best_position;       // Best position found
} EFOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel generate_particle_kernel; // Kernel for generating new particles
    cl_bool owns_queue;         // Tracks if queue was created locally
} EFOCLContext;

// Function prototypes
void EFO_init_cl(EFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void EFO_cleanup_cl(EFOCLContext *cl_ctx);
void EFO_init_context(EFOContext *ctx, Optimizer *opt, EFOCLContext *cl_ctx);
void EFO_cleanup_context(EFOContext *ctx, EFOCLContext *cl_ctx);
void EFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
