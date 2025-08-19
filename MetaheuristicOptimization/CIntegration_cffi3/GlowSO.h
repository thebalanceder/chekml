#ifndef GLOWSO_H
#define GLOWSO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define GSO_LUCIFERIN_INITIAL 5.0f
#define GSO_DECISION_RANGE_INITIAL 3.0f
#define GSO_LUCIFERIN_DECAY 0.4f
#define GSO_LUCIFERIN_ENHANCEMENT 0.6f
#define GSO_NEIGHBOR_THRESHOLD 0.08f
#define GSO_STEP_SIZE 0.6f
#define GSO_SENSOR_RANGE 10.0f
#define GSO_NEIGHBOR_COUNT 10
#define GSO_POPULATION_SIZE 50
#define GSO_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Glowworm positions
    cl_mem fitness;             // Luciferin values (fitness)
    cl_mem decision_range;      // Decision ranges for each glowworm
    cl_mem distances;           // Pairwise distances
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each glowworm
    cl_mem best_position;       // Best position found
    cl_float best_fitness;      // Best fitness value
} GlowSOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel luciferin_kernel; // Kernel for luciferin update
    cl_kernel movement_kernel;  // Kernel for movement phase
    cl_kernel range_kernel;     // Kernel for decision range update
    cl_bool owns_queue;         // Tracks if queue was created locally
} GlowSOCLContext;

// Function prototypes
void GlowSO_init_cl(GlowSOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void GlowSO_cleanup_cl(GlowSOCLContext *cl_ctx);
void GlowSO_init_context(GlowSOContext *ctx, Optimizer *opt, GlowSOCLContext *cl_ctx);
void GlowSO_cleanup_context(GlowSOContext *ctx, GlowSOCLContext *cl_ctx);
void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
