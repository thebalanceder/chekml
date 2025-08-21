#ifndef SGO_H
#define SGO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>  // ✅ AVX2 intrinsics for vectorization
#include <math.h>
#include <stdlib.h>     // ✅ Ensure malloc/free work properly
#include <stdio.h>      // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define ATTACK_RATE 0.5
#define DEFENSE_STRENGTH 0.3
#define FIGHT_INTENSITY 0.2
#define WIN_THRESHOLD 0.6
#define RAND_BUFFER_SIZE 65536  // ✅ Precomputed random number buffer size

// ⚙️ SGO Algorithm Phases
void sgo_initialize_players(Optimizer *opt, double *rand_buffer, int *rand_idx);
void sgo_divide_teams(Optimizer *opt, int *offensive_indices, int *defensive_indices, int *offensive_size);
void sgo_simulate_fights(Optimizer *opt, int *offensive_indices, int *defensive_indices, int size, double *rand_buffer, int *rand_idx);
void sgo_determine_winners(Optimizer *opt, int *offensive_indices, int offensive_size, int *defensive_indices, int defensive_size, int *winners, int *winner_count);
void sgo_update_positions(Optimizer *opt, int *winners, int winner_count, double *rand_buffer, int *rand_idx);
void SGO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SGO_H
