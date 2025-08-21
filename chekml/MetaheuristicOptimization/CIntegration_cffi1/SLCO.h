#ifndef SLCO_H
#define SLCO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h> // For uint64_t
#include <immintrin.h> // For SIMD intrinsics
#include "generaloptimizer.h"

// 🔧 SLCO Parameters
#define SLCO_ALPHA 2.0
#define SLCO_BETA 2.0
#define SLCO_MUTATION_PROB 0.02
#define SLCO_N_TEAMS 10
#define SLCO_N_MAIN_PLAYERS 5
#define SLCO_N_RESERVE_PLAYERS 3
#define SLCO_TOTAL_PLAYERS_PER_TEAM (SLCO_N_MAIN_PLAYERS + SLCO_N_RESERVE_PLAYERS)

// ⚙️ Struct Definitions
typedef struct {
    double *positions; // Contiguous block for all player positions
    double *costs;     // Contiguous block for all player costs
    double total_cost; // Inverse of mean main player cost
} Team;

typedef struct {
    Team *teams;          // Array of teams
    int n_teams;          // Number of teams
    int n_main_players;   // Number of main players per team
    int n_reserve_players;// Number of reserve players per team
    int best_team_idx;    // Index of the best team
    double best_total_cost;// Best total cost
} League;

typedef struct {
    Optimizer base;           // Base optimizer struct
    League league;           // SLCO league structure
    ObjectiveFunction objective_function; // Objective function pointer
    double *temp_buffer;     // Reusable buffer for calculations
    double *all_positions;   // Contiguous buffer for all player positions
    double *all_costs;       // Contiguous buffer for all player costs
    uint64_t rng_state;      // Xorshift RNG state
} SLCO_Optimizer;

// 🌊 Utility Functions
void enforce_bound_constraints(Optimizer *opt);
static inline double slco_fast_rand(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return ((double)(*state * 0x2545F4914F6CDD1DULL) / 1.8446744073709551616e19); // 2^64
}

// 🌊 SLCO Algorithm Phases
void initialize_league(SLCO_Optimizer *opt);
void takhsis(SLCO_Optimizer *opt);
void update_total_cost(SLCO_Optimizer *opt);
void probability_host(SLCO_Optimizer *opt, int ii, int jj, int *winner, int *loser);
void winner_function_main(SLCO_Optimizer *opt);
void winner_function_reserve(SLCO_Optimizer *opt);
void loser_function(SLCO_Optimizer *opt);
void competition(SLCO_Optimizer *opt, int iteration);
void free_league(SLCO_Optimizer *opt);

// 🚀 Optimization Execution
void SLCO_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SLCO_H
