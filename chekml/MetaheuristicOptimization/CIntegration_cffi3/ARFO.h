#ifndef ARFO_H
#define ARFO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define ARFO_BRANCHING_THRESHOLD 0.6f
#define ARFO_MAX_BRANCHING 5
#define ARFO_MIN_BRANCHING 1
#define ARFO_INITIAL_STD 1.0f
#define ARFO_FINAL_STD 0.01f
#define ARFO_MAX_ELONGATION 0.1f
#define ARFO_LOCAL_INERTIA 0.5f
#define ARFO_ELIMINATION_PERCENTILE 10.0f
#define ARFO_AUXIN_NORMALIZATION_FACTOR 1.0f
#define ARFO_MAX_EVALS_DEFAULT 10000

// Struct for fitness sorting
typedef struct {
    cl_float fitness;
    cl_int index;
} ARFOFitnessIndex;

// ARFO context for GPU
typedef struct {
    cl_mem population;          // Particle positions
    cl_mem fitness;             // Fitness values
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each particle
    cl_mem auxin;               // Auxin concentrations
    cl_mem auxin_sorted;        // Sorted auxin concentrations
    cl_mem topology;            // Von Neumann topology
    cl_mem new_roots;           // New roots for branching
    cl_mem fitness_indices;     // Fitness indices for sorting
    cl_float best_fitness;      // Best fitness value
    cl_mem best_position;       // Best position found
    cl_mem new_root_count;      // Counter for new roots
    cl_mem population_size;     // Current population size
    cl_int original_pop_size;   // Original population size
} ARFOContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel regrowth_kernel;  // Kernel for regrowth phase
    cl_kernel branching_kernel; // Kernel for branching phase
    cl_kernel lateral_growth_kernel; // Kernel for lateral growth phase
    cl_kernel elimination_kernel;   // Kernel for elimination phase
    cl_kernel replenish_kernel;     // Kernel for replenishment phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} ARFOCLContext;

// Function prototypes
void ARFO_init_cl(ARFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void ARFO_cleanup_cl(ARFOCLContext *cl_ctx);
void ARFO_init_context(ARFOContext *ctx, Optimizer *opt, ARFOCLContext *cl_ctx);
void ARFO_cleanup_context(ARFOContext *ctx, ARFOCLContext *cl_ctx);
void ARFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
