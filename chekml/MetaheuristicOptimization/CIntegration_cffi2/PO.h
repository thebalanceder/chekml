#ifndef POLITICAL_OPTIMIZER_H
#define POLITICAL_OPTIMIZER_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define PARTIES 8
#define LAMBDA_RATE 1.0

// Political Optimizer Algorithm Phases
void election_phase(Optimizer *opt, double *fitness, double *a_winners, int *a_winner_indices);
void government_formation_phase(Optimizer *opt, double *fitness, ObjectiveFunction objective_function, double *temp_pos);
void election_campaign_phase(Optimizer *opt, double *fitness, double *prev_positions, ObjectiveFunction objective_function, double *temp_pos);
void party_switching_phase(Optimizer *opt, double *fitness, int t);
void parliamentarism_phase(Optimizer *opt, double *fitness, double *a_winners, int *a_winner_indices, ObjectiveFunction objective_function, double *temp_pos);

// Optimization Execution
void PO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // POLITICAL_OPTIMIZER_H
