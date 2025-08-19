#ifndef PRO_H
#define PRO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define PRO_POPULATION_SIZE 30
#define PRO_MAX_EVALUATIONS 50
#define REINFORCEMENT_RATE 0.7f
#define SCHEDULE_MIN 0.9f
#define SCHEDULE_MAX 1.0f

typedef struct {
    cl_mem population;          // Population positions
    cl_mem fitness;             // Current fitness values
    cl_mem schedules;          // Schedule values for each individual
    cl_mem best_position;      // Best position found
    cl_mem bounds;             // Problem bounds
    cl_mem random_seeds;       // Random seeds for each individual
    cl_float best_fitness;     // Best fitness value
} PROContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel stimulate_kernel; // Kernel for behavior stimulation
    cl_kernel reschedule_kernel;// Kernel for rescheduling
    cl_bool owns_queue;         // Tracks if queue was created locally
} PROCLContext;

// Function prototypes
void PRO_init_cl(PROCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void PRO_cleanup_cl(PROCLContext *cl_ctx);
void PRO_init_context(PROContext *ctx, Optimizer *opt, PROCLContext *cl_ctx);
void PRO_cleanup_context(PROContext *ctx, PROCLContext *cl_ctx);
void PRO_optimize(Optimizer *opt, double (*objective_function)(double *));
void quicksort_with_indices_pro(float *arr, int *indices, int low, int high); // Updated to float *
void select_behaviors(Optimizer *opt, int i, int current_eval, float *schedules, int *selected_behaviors, int *landa);
void apply_reinforcement(Optimizer *opt, int i, int *selected_behaviors, int landa, float *schedules, float *new_solution, float new_fitness, PROContext *ctx, PROCLContext *cl_ctx);

#endif
