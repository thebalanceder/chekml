#ifndef WPA_H
#define WPA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization Parameters
#define WPA_R1_MIN 0.0f
#define WPA_R1_MAX 2.0f
#define WPA_R2_MIN 0.0f
#define WPA_R2_MAX 1.0f
#define WPA_R3_MIN 0.0f
#define WPA_R3_MAX 2.0f
#define WPA_F_MIN -5.0f
#define WPA_F_MAX 5.0f
#define WPA_C_MIN -5.0f
#define WPA_C_MAX 5.0f
#define WPA_K_INITIAL 1.0f
#define WPA_STAGNATION_THRESHOLD 3
#define WPA_POPULATION_SIZE 20
#define WPA_MAX_ITER 500

// Algorithm Constants
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

typedef struct {
    cl_mem population;          // Particle positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each particle
    cl_mem stagnation_counts;   // Stagnation counters
    cl_mem mu_P;                // Population mean per dimension
    cl_mem sigma;               // Population std dev per dimension
    cl_float best_fitness;      // Best fitness value
} WPAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel exploration_kernel; // Kernel for exploration phase
    cl_kernel exploitation_kernel; // Kernel for exploitation phase
    cl_kernel stats_kernel;     // Kernel for computing stats
    cl_bool owns_queue;         // Tracks if queue was created locally
} WPACLContext;

// Function prototypes
void WPA_init_cl(WPACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void WPA_cleanup_cl(WPACLContext *ctx);
void WPA_init_context(WPAContext *ctx, Optimizer *opt, WPACLContext *cl_ctx);
void WPA_cleanup_context(WPAContext *ctx, WPACLContext *cl_ctx);
void WPA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif // WPA_H
