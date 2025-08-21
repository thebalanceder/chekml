#ifndef FLYFO_H
#define FLYFO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

#define SURVIVAL_LIST_RATIO 0.2f

typedef struct {
    cl_mem population;
    cl_mem fitness;
    cl_mem past_fitness;
    cl_mem survival_list;
    cl_mem survival_fitness;
    cl_mem best_position;
    cl_mem bounds;
    cl_mem random_seeds;
    cl_float best_fitness;
    cl_float worst_fitness;
    cl_int survival_count;
} FlyFOContext;

typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_pop_kernel;
    cl_kernel fuzzy_tuning_kernel;
    cl_kernel update_position_kernel;
    cl_kernel crossover_kernel;
    cl_kernel suffocation_kernel;
    cl_bool owns_queue; // Tracks if queue was created locally
} FlyFOCLContext;

void FlyFO_init_cl(FlyFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void FlyFO_cleanup_cl(FlyFOCLContext *cl_ctx);
void FlyFO_init_context(FlyFOContext *ctx, Optimizer *opt, FlyFOCLContext *cl_ctx);
void FlyFO_cleanup_context(FlyFOContext *ctx, FlyFOCLContext *cl_ctx);
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
