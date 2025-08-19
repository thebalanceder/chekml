#ifndef SLCO_H
#define SLCO_H

#include <CL/cl.h>
#include "generaloptimizer.h"

// SLCO Parameters
#define SLCO_ALPHA_MAX 2.0f
#define SLCO_ALPHA_MIN 0.5f
#define SLCO_BETA_MAX 2.0f
#define SLCO_BETA_MIN 0.5f
#define SLCO_MUTATION_PROB 0.05f
#define SLCO_S_SAY 0.5f
#define SLCO_N_TEAMS 10
#define SLCO_N_MAIN_PLAYERS 5
#define SLCO_N_RESERVE_PLAYERS 3
#define SLCO_STAGNATION_THRESHOLD 50
#define SLCO_CONVERGENCE_TOL 1e-6f
#define SLCO_MAX_EVALS_DEFAULT 100

// Struct Definitions
typedef struct {
    cl_mem positions;           // Player positions
    cl_mem costs;              // Player costs
    cl_mem best_position;      // Best position found
    cl_float best_cost;        // Best cost found
    cl_mem bounds;             // Problem bounds
    cl_mem random_seeds;       // Random seeds for each player
    cl_mem teams;              // Team assignments (main/reserve)
    cl_mem team_total_costs;   // Total cost per team
    cl_int best_team_idx;      // Index of the best team
} SLCOContext;

typedef struct {
    cl_context context;        // OpenCL context
    cl_device_id device;       // OpenCL device
    cl_command_queue queue;    // OpenCL command queue
    cl_program program;        // OpenCL program
    cl_kernel init_league_kernel;  // Kernel for league initialization
    cl_kernel takhsis_kernel;      // Kernel for player reassignment
    cl_kernel winner_main_kernel;  // Kernel for winner main player update
    cl_kernel winner_reserve_kernel; // Kernel for winner reserve player update
    cl_kernel loser_kernel;        // Kernel for loser team update
    cl_bool owns_queue;           // Tracks if queue was created locally
} SLCOCLContext;

// Function Prototypes
void SLCO_init_cl(SLCOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt);
void SLCO_cleanup_cl(SLCOCLContext *cl_ctx);
void SLCO_init_context(SLCOContext *ctx, Optimizer *opt, SLCOCLContext *cl_ctx);
void SLCO_cleanup_context(SLCOContext *ctx, SLCOCLContext *cl_ctx);
void SLCO_optimize(Optimizer *opt, double (*objective_function)(double *));

#endif
