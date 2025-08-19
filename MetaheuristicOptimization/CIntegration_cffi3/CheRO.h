#ifndef CHERO_H
#define CHERO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CheRO_INITIAL_KE 1000.0f
#define CheRO_MOLE_COLL 0.5f
#define CheRO_BUFFER_INITIAL 0.0f
#define CheRO_ALPHA 10.0f
#define CheRO_BETA 0.2f
#define CheRO_SPLIT_RATIO 0.5f
#define CheRO_ELIMINATION_RATIO 0.2f
#define CheRO_POPULATION_SIZE 100
#define CheRO_MAX_ITER 500
#define CheRO_MAX_POPULATION 1000

// Default maximum evaluations if not specified
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif

// Molecule structure for GPU
typedef struct {
    cl_mem position;            // Molecule positions
    cl_mem pe;                  // Potential energy (fitness)
    cl_mem ke;                  // Kinetic energy
    cl_mem bounds;              // Problem bounds
    cl_mem random_seeds;        // Random seeds for each molecule
    cl_float best_fitness;      // Best fitness value
    cl_mem best_position;       // Best position found
} CheROContext;

typedef struct {
    cl_context context;         // OpenCL context
    cl_device_id device;        // OpenCL device
    cl_command_queue queue;     // OpenCL command queue
    cl_program program;         // OpenCL program
    cl_kernel init_pop_kernel;  // Kernel for population initialization
    cl_kernel on_wall_kernel;   // Kernel for on-wall collision
    cl_kernel decomp_kernel;    // Kernel for decomposition
    cl_kernel inter_coll_kernel;// Kernel for inter-molecular collision
    cl_kernel synthesis_kernel; // Kernel for synthesis
    cl_kernel elim_kernel;      // Kernel for elimination
    cl_bool owns_queue;         // Tracks if queue was created locally
} CheROCLContext;

// Function prototypes
void CheRO_init_cl(CheROCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void CheRO_cleanup_cl(CheROCLContext *cl_ctx);
void CheRO_init_context(CheROContext *ctx, Optimizer *opt, CheROCLContext *cl_ctx);
void CheRO_cleanup_context(CheROContext *ctx, CheROCLContext *cl_ctx);
void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
