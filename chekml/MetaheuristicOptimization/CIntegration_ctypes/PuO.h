#ifndef PUO_H
#define PUO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization Parameters
#define Q_PROBABILITY 0.67
#define BETA_FACTOR 2.0
#define PF1 0.5
#define PF2 0.5
#define PF3 0.3
#define MEGA_EXPLORE_INIT 0.99
#define MEGA_EXPLOIT_INIT 0.99
#define PCR_INITIAL 0.80
#define POPULATION_SIZE_DEFAULT 30
#define MAX_ITER_DEFAULT 500

// Puma Optimizer State
typedef struct {
    double Q;
    double beta;
    double PF[3];
    double mega_explore;
    double mega_exploit;
    double unselected[2];
    double f3_explore;
    double f3_exploit;
    double seq_time_explore[3];
    double seq_time_exploit[3];
    double seq_cost_explore[3];
    double seq_cost_exploit[3];
    double score_explore;
    double score_exploit;
    double *pf_f3;
    int pf_f3_size;
    int flag_change;
} PumaOptimizerState;

// Puma Optimizer Algorithm Phases
void initialize_pumas(Optimizer *opt);
void puma_exploration_phase(Optimizer *opt, PumaOptimizerState *state, double (*objective_function)(double *));
void puma_exploitation_phase(Optimizer *opt, PumaOptimizerState *state, int iter);
void evaluate_pumas(Optimizer *opt, double (*objective_function)(double *));
void enforce_puma_bounds(Optimizer *opt);

// Optimization Execution
void PuO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // PUO_H
