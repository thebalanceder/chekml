#ifndef MSO_H
#define MSO_H
#include <CL/cl.h>
#include "generaloptimizer.h"
#define MSO_P_EXPLORE 0.2f
#define MSO_MAX_P_EXPLORE 0.8f
#define MSO_MIN_P_EXPLORE 0.1f
#define MSO_PERTURBATION_SCALE 10.0f
#define MSO_POPULATION_SIZE 20
#define MSO_MAX_ITER 500
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 10000
#endif
typedef struct {
    cl_mem population;
    cl_mem fitness;
    cl_mem best_position;
    cl_mem bounds;
    cl_mem random_seeds;
    cl_float best_fitness;
} MSOContext;
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel update_kernel;
    cl_bool owns_queue;
} MSOCLContext;
void MSO_init_cl(MSOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void MSO_cleanup_cl(MSOCLContext *cl_ctx);
void MSO_init_context(MSOContext *ctx, Optimizer *opt, MSOCLContext *cl_ctx);
void MSO_cleanup_context(MSOContext *ctx, MSOCLContext *cl_ctx);
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *));
#endif
