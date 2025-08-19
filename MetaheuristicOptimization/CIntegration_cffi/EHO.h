#ifndef EHO_H
#define EHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define EHO_ALPHA 0.5
#define EHO_BETA 0.1
#define EHO_KEEP 2
#define EHO_NUM_CLANS 5

// EHO Algorithm Phases
void eho_initialize_population(Optimizer *opt, ObjectiveFunction objective_function);
void eho_clan_division_phase(Optimizer *opt);
void eho_clan_updating_phase(Optimizer *opt);
void eho_separating_phase(Optimizer *opt);
void eho_elitism_phase(Optimizer *opt);
void eho_update_best_solution(Optimizer *opt, ObjectiveFunction objective_function);

// Main Optimization Function
// Matches void (*optimize)(void*, ObjectiveFunction) where ObjectiveFunction is double (*)(double*)
void EHO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // EHO_H
