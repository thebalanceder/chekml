#ifndef SAO_H
#define SAO_H
#include <CL/cl.h>
#include "generaloptimizer.h"
// Optimization Parameters
#define SAO_NUM_POP 50
#define SAO_MAX_ITER 1000
#define SAO_LOW -5.0f
#define SAO_UP 5.0f
#define SAO_DF_MIN 0.35f
#define SAO_DF_MAX 0.6f
#define SAO_NUM_ELITES 4
#define SAO_BROWNIAN_VARIANCE 0.5f
#define SAO_EULER 2.718281828459045f
// Default maximum evaluations if not specified
#ifndef SAO_MAX_EVALS_DEFAULT
#define SAO_MAX_EVALS_DEFAULT 100
#endif
typedef struct {
    cl_mem population;
    cl_mem fitness;
    cl_mem best_position;
    cl_mem bounds;
    cl_mem random_seeds;
    cl_mem centroid;
    cl_mem elite;
    cl_mem brownian;
    cl_mem qq;
    cl_mem reverse_pop;
    cl_float best_fitness;
} SAOContext;
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel brownian_kernel;
    cl_kernel centroid_kernel;
    cl_kernel exploration_kernel;
    cl_kernel development_kernel;
    cl_kernel reverse_learning_kernel;
    cl_bool owns_queue;
} SAOCLContext;
// Function prototypes
void SAO_init_cl(SAOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SAO_cleanup_cl(SAOCLContext *cl_ctx);
void SAO_init_context(SAOContext *ctx, Optimizer *opt, SAOCLContext *cl_ctx);
void SAO_cleanup_context(SAOContext *ctx, SAOCLContext *cl_ctx);
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *));
#endif
