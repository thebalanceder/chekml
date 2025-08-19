#ifndef PUO_H
#define PUO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization Parameters
#define PUO_Q_PROBABILITY 0.67f
#define PUO_BETA_FACTOR 2.0f
#define PUO_PF1 0.5f
#define PUO_PF2 0.5f
#define PUO_PF3 0.3f
#define PUO_MEGA_EXPLORE_INIT 0.99f
#define PUO_MEGA_EXPLOIT_INIT 0.99f
#define PUO_PCR_INITIAL 0.80f
#define PUO_POPULATION_SIZE_DEFAULT 30
#define PUO_MAX_ITER_DEFAULT 500
#define PUO_PF_F3_MAX_SIZE 6

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif

// Puma Optimizer GPU Context
typedef struct {
    cl_mem population;          // Puma positions
    cl_mem fitness;             // Current fitness values
    cl_mem best_position;       // Best position found
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each puma
    cl_float best_fitness;      // Best fitness value
    // Buffers for exploration phase
    cl_mem temp_position;       // Temporary position storage
    cl_mem y;                   // Intermediate vector y
    cl_mem z;                   // Intermediate vector z
    // Buffers for exploitation phase
    cl_mem beta2;               // Beta2 vector
    cl_mem w;                   // W vector
    cl_mem v;                   // V vector
    cl_mem F1;                  // F1 vector
    cl_mem F2;                  // F2 vector
    cl_mem S1;                  // S1 vector
    cl_mem S2;                  // S2 vector
    cl_mem VEC;                 // VEC vector
    cl_mem Xatack;              // Xatack vector
    cl_mem mbest;               // Mean best position
} PuOContext;

// Puma Optimizer OpenCL Context
typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel explore_kernel;   // Kernel for exploration phase
    cl_kernel exploit_kernel;   // Kernel for exploitation phase
    cl_bool owns_queue;         // Tracks if queue was created locally
} PuOCLContext;

// Puma Optimizer State for CPU-side tracking
typedef struct {
    float q_probability;
    float beta;
    float PF[3];
    float mega_explore;
    float mega_exploit;
    float unselected[2];
    float f3_explore;
    float f3_exploit;
    float seq_time_explore[3];
    float seq_time_exploit[3];
    float seq_cost_explore[3];
    float seq_cost_exploit[3];
    float score_explore;
    float score_exploit;
    float pf_f3[PUO_PF_F3_MAX_SIZE];
    int pf_f3_size;
    int flag_change;
} PuOState;

// Function prototypes
void PuO_init_cl(PuOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void PuO_cleanup_cl(PuOCLContext *cl_ctx);
void PuO_init_context(PuOContext *ctx, Optimizer *opt, PuOCLContext *cl_ctx);
void PuO_cleanup_context(PuOContext *ctx, PuOCLContext *cl_ctx);
void PuO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
