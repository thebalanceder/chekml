#ifndef SGO_H
#define SGO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define ATTACK_RATE 0.5
#define DEFENSE_STRENGTH 0.3
#define FIGHT_INTENSITY 0.2
#define WIN_THRESHOLD 0.6

// ⚙️ SGO Algorithm Phases
void initialize_players(Optimizer *opt);
void divide_teams(Optimizer *opt, int *offensive_indices, int *defensive_indices, int *offensive_size);
void simulate_fight(Optimizer *opt, int offensive_idx, int defensive_idx);
void determine_winners(Optimizer *opt, int *offensive_indices, int offensive_size, int *defensive_indices, int defensive_size, int *winners, int *winner_count);
void update_positions_sgo(Optimizer *opt, int *winners, int winner_count);

// 🚀 Optimization Execution
void SGO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SGO_H
