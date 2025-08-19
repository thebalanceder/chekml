#ifndef PUO_H
#define PUO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

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
#define PF_F3_MAX_SIZE 6

// Puma Optimizer State
typedef struct {
    double q_probability; // Renamed from Q to avoid macro conflict
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
    double pf_f3[PF_F3_MAX_SIZE];
    int pf_f3_size;
    int flag_change;
    // Preallocated buffers for exploration phase
    double *temp_position;
    double *y;
    double *z;
    double *fitness;
    int *indices;
    int *perm;
    // Preallocated buffers for exploitation phase
    double *beta2;
    double *w;
    double *v;
    double *F1;
    double *F2;
    double *S1;
    double *S2;
    double *VEC;
    double *Xatack;
    double *mbest;
} PumaOptimizerState;

// Puma Optimizer Algorithm Phases
void initialize_pumas(Optimizer *opt, PumaOptimizerState *state);
void puma_exploration_phase(Optimizer *opt, PumaOptimizerState *state, double (*objective_function)(double *));
void puma_exploitation_phase(Optimizer *opt, PumaOptimizerState *state, int iter);
void evaluate_pumas(Optimizer *opt, double (*objective_function)(double *));
void enforce_puma_bounds(Optimizer *opt);
void PuO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // PUO_H
