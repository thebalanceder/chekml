#ifndef LOA_H
#define LOA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define LOA_NOMAD_RATIO 0.2f
#define LOA_PRIDE_SIZE 5
#define LOA_FEMALE_RATIO 0.8f
#define LOA_ROAMING_RATIO 0.2f
#define LOA_MATING_RATIO 0.2f
#define LOA_MUTATION_PROB 0.1f
#define LOA_IMMIGRATION_RATIO 0.1f
#define LOA_POPULATION_SIZE 20
#define LOA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem population;          // Lion positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each lion
    cl_mem prides;              // Pride assignments (lion indices)
    cl_mem pride_sizes;         // Number of lions per pride
    cl_mem nomads;              // Nomad lion indices
    cl_mem genders;             // 1 for female, 0 for male
    cl_mem temp_buffer;         // Temporary buffer for prey, offspring, etc.
    cl_mem females;             // Female indices per pride
    cl_mem hunters;             // Hunter indices per pride
    cl_mem non_hunters;         // Non-hunter indices per pride
    cl_mem males;               // Male indices per pride
    cl_mem mating_females;      // Females selected for mating
    cl_mem nomad_females;       // Female nomads
    cl_int num_prides;          // Number of prides
    cl_int nomad_size;          // Number of nomads
    cl_float best_fitness;      // Best fitness value
} LOAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel hunting_kernel;   // Kernel for hunting phase
    cl_kernel move_safe_kernel; // Kernel for move to safe place phase
    cl_kernel roaming_kernel;   // Kernel for roaming phase
    cl_kernel mating_kernel;    // Kernel for mating phase
    cl_kernel nomad_move_kernel;// Kernel for nomad movement
    cl_kernel pop_control_kernel;// Kernel for population control
    cl_bool owns_queue;         // Tracks if queue was created locally
} LOACLContext;

// Function prototypes
void LOA_init_cl(LOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void LOA_cleanup_cl(LOACLContext *cl_ctx);
void LOA_init_context(LOAContext *ctx, Optimizer *opt, LOACLContext *cl_ctx);
void LOA_cleanup_context(LOAContext *ctx, LOACLContext *cl_ctx);
void LOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
