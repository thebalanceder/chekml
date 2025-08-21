#ifndef MFA_H
#define MFA_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MFA_INERTIA_WEIGHT 0.8f
#define MFA_INERTIA_DAMP 1.0f
#define MFA_PERSONAL_COEFF 1.0f
#define MFA_GLOBAL_COEFF1 1.5f
#define MFA_GLOBAL_COEFF2 1.5f
#define MFA_DISTANCE_COEFF 2.0f
#define MFA_NUPTIAL_DANCE 5.0f
#define MFA_RANDOM_FLIGHT 1.0f
#define MFA_DANCE_DAMP 0.8f
#define MFA_FLIGHT_DAMP 0.99f
#define MFA_NUM_OFFSPRING 20
#define MFA_NUM_MUTANTS 1
#define MFA_MUTATION_RATE 0.01f
#define MFA_POPULATION_SIZE 40
#define MFA_MAX_ITER 500

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

typedef struct {
    cl_mem male_population;     // Male positions
    cl_mem female_population;   // Female positions
    cl_mem male_fitness;        // Male fitness values
    cl_mem female_fitness;      // Female fitness values
    cl_mem male_velocities;     // Male velocities
    cl_mem female_velocities;   // Female velocities
    cl_mem best_male_positions; // Best positions for males
    cl_mem best_position;       // Global best position
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each individual
    cl_float best_fitness;      // Best fitness value
} MfAContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel update_males_kernel; // Kernel for updating males
    cl_kernel update_females_kernel; // Kernel for updating females
    cl_kernel mating_kernel;    // Kernel for mating phase
    cl_kernel mutation_kernel;  // Kernel for mutation phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} MfACLContext;

// Function prototypes
void MfA_init_cl(MfACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void MfA_cleanup_cl(MfACLContext *cl_ctx);
void MfA_init_context(MfAContext *ctx, Optimizer *opt, MfACLContext *cl_ctx);
void MfA_cleanup_context(MfAContext *ctx, MfACLContext *cl_ctx);
void MfA_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
