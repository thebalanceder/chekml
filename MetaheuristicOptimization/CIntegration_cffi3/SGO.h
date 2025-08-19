#ifndef SGO_H
#define SGO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SGO_ATTACK_RATE 0.5f
#define SGO_DEFENSE_STRENGTH 0.3f
#define SGO_FIGHT_INTENSITY 0.2f
#define SGO_WIN_THRESHOLD 0.6f
#define SGO_MAX_EVALS_DEFAULT 10000
#define SGO_CONVERGENCE_TOL 1e-6
#define SGO_STAGNATION_THRESHOLD 10

// SGO Context for GPU
typedef struct {
    cl_mem positions;       // Population positions
    cl_mem costs;           // Fitness values
    cl_mem best_position;   // Best solution position
    cl_mem bounds;          // Bounds for each dimension
    cl_mem random_seeds;    // Random seeds for GPU
    cl_mem team_assignments;// Offensive/defensive team indices
    cl_mem winners;         // Winner indices
    double best_cost;       // Best fitness value
    int offensive_size;     // Number of offensive players
} SGOContext;

// SGO OpenCL Context
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_players_kernel;
    cl_kernel divide_teams_kernel;
    cl_kernel simulate_fight_kernel;
    cl_kernel determine_winners_kernel;
    cl_kernel update_positions_kernel;
    cl_bool owns_queue;
} SGOCLContext;

// Function Prototypes
void SGO_init_cl(SGOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SGO_cleanup_cl(SGOCLContext *cl_ctx);
void SGO_init_context(SGOContext *ctx, Optimizer *opt, SGOCLContext *cl_ctx);
void SGO_cleanup_context(SGOContext *ctx, SGOCLContext *cl_ctx);
void SGO_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SGO_H
