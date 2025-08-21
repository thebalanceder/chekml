#ifndef JAGUAR_ALGORITHM_H
#define JAGUAR_ALGORITHM_H
#include <CL/cl.h>
#include "generaloptimizer.h"
#define JA_CRUISING_PROBABILITY 0.8f
#define JA_CRUISING_DISTANCE 0.1f
#define JA_ALPHA 0.1f
#define JA_POPULATION_SIZE 50
#define JA_MAX_ITER 500
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
} JAContext;
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel cruising_kernel;
    cl_kernel random_walk_kernel;
    cl_bool owns_queue;
} JACLContext;
void JA_init_cl(JACLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void JA_cleanup_cl(JACLContext *cl_ctx);
void JA_init_context(JAContext *ctx, Optimizer *opt, JACLContext *cl_ctx);
void JA_cleanup_context(JAContext *ctx, JACLContext *cl_ctx);
void JA_optimize(Optimizer *opt, double (*objective_function)(double *));
#endif // JAGUAR_ALGORITHM_H
