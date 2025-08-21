#ifndef SLCO_H
#define SLCO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 SLCO Parameters
#define SLCO_ALPHA_MAX 2.0
#define SLCO_ALPHA_MIN 0.5
#define SLCO_BETA_MAX 2.0
#define SLCO_BETA_MIN 0.5
#define SLCO_MUTATION_PROB 0.05
#define SLCO_S_SAY 0.5
#define SLCO_N_TEAMS 10
#define SLCO_N_MAIN_PLAYERS 5
#define SLCO_N_RESERVE_PLAYERS 3
#define SLCO_STAGNATION_THRESHOLD 50
#define SLCO_CONVERGENCE_TOL 1e-6

// ⚙️ Struct Definitions
typedef struct {
    double *position;  // Position in the search space (dim-dimensional)
    double cost;       // Objective function value
} Player;

typedef struct {
    Player *main_players;    // Array of main players
    Player *reserve_players; // Array of reserve players
    double total_cost;       // Inverse of mean main player cost
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
    Player *all_players;     // Preallocated buffer for all players
    int stagnation_count;    // Counter for convergence check
    double prev_best_cost;   // Previous best cost for convergence
} SLCO_Optimizer;

// 🌊 Utility Functions
void enforce_bound_constraints(Optimizer *opt);

// 🌊 SLCO Algorithm Phases
void initialize_league(SLCO_Optimizer *opt);
void takhsis(SLCO_Optimizer *opt);
void update_total_cost(SLCO_Optimizer *opt);
void probability_host(League *league, int ii, int jj, int *winner, int *loser);
void winner_function_main(SLCO_Optimizer *opt, int iteration);
void winner_function_reserve(SLCO_Optimizer *opt, int iteration);
void loser_function(SLCO_Optimizer *opt, int iteration);
void competition(SLCO_Optimizer *opt, int iteration);
void free_league(SLCO_Optimizer *opt);

// 🚀 Optimization Execution
void SLCO_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SLCO_H
