#ifndef SCA_H
#define SCA_H
#include <CL/cl.h>
#include "generaloptimizer.h"
#define SCA_A 2.0f
#define SCA_POPULATION_SIZE 20
#define SCA_MAX_ITER 500
#ifndef MAX_EVALS_DEFAULT
#define MAX_EVALS_DEFAULT 100
#endif
typedef struct {
    cl_mem population;
    cl_mem fitness;
    cl_mem best_position;
    cl_mem bounds;
    cl_mem random_seeds;
    cl_float best_fitness;
} SCAContext;
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel update_kernel;
    cl_bool owns_queue;
} SCACLContext;
void SCA_init_cl(SCACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SCA_cleanup_cl(SCACLContext *cl_ctx);
void SCA_init_context(SCAContext *ctx, Optimizer *opt, SCACLContext *cl_ctx);
void SCA_cleanup_context(SCAContext *ctx, SCACLContext *cl_ctx);
void SCA_optimize(Optimizer *opt, double (*objective_function)(double *));
#endif
