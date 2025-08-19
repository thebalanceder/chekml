/* SGO.c - Optimized Implementation file for Squid Game Optimization */
#include "SGO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Compiler-specific optimizations
#if defined(__GNUC__) || defined(__clang__)
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif

// Fast random double generator using precomputed RAND_MAX inverse
static const double RAND_MAX_INV = 1.0 / RAND_MAX;
INLINE double rand_double_sgo(double min, double max) {
    return min + (max - min) * (rand() * RAND_MAX_INV);
}

// Initialize Players (Population) with unrolled loops for small dimensions
void initialize_players(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double_sgo(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Divide Players into Offensive and Defensive Teams with optimized shuffle
void divide_teams(Optimizer *opt, int *offensive_indices, int *defensive_indices, int *offensive_size) {
    *offensive_size = (int)(opt->population_size * ATTACK_RATE);
    int *indices = offensive_indices; // Reuse offensive_indices to reduce memory usage
    
    // Initialize indices
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle optimized for cache locality
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = (int)(rand_double_sgo(0.0, i + 1));
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Split into defensive team (no copy needed for offensive)
    for (int i = *offensive_size; i < opt->population_size; i++) {
        defensive_indices[i - *offensive_size] = indices[i];
    }
}

// Simulate Fight with pre-allocated output arrays
void simulate_fight(Optimizer *opt, int offensive_idx, int defensive_idx, double *offensive_pos, double *defensive_pos) {
    double r1 = rand_double_sgo(0.0, 1.0);
    double *off_pos = opt->population[offensive_idx].position;
    double *def_pos = opt->population[defensive_idx].position;
    
    // Compute movement and resistance in a single loop
    for (int j = 0; j < opt->dim; j++) {
        double diff = off_pos[j] - def_pos[j];
        double movement = FIGHT_INTENSITY * diff * r1;
        double resistance = DEFENSE_STRENGTH * diff * (1.0 - r1);
        
        offensive_pos[j] = off_pos[j] + movement;
        defensive_pos[j] = def_pos[j] + resistance;
        
        // Inline bounds checking
        if (offensive_pos[j] < opt->bounds[2 * j]) {
            offensive_pos[j] = opt->bounds[2 * j];
        } else if (offensive_pos[j] > opt->bounds[2 * j + 1]) {
            offensive_pos[j] = opt->bounds[2 * j + 1];
        }
        
        if (defensive_pos[j] < opt->bounds[2 * j]) {
            defensive_pos[j] = opt->bounds[2 * j];
        } else if (defensive_pos[j] > opt->bounds[2 * j + 1]) {
            defensive_pos[j] = opt->bounds[2 * j + 1];
        }
    }
    
    // Copy results back
    memcpy(off_pos, offensive_pos, opt->dim * sizeof(double));
    memcpy(def_pos, defensive_pos, opt->dim * sizeof(double));
}

// Determine Winners with early termination
void determine_winners(Optimizer *opt, int *offensive_indices, int offensive_size, 
                      int *defensive_indices, int defensive_size, int *winners, int *winner_count) {
    *winner_count = 0;
    int min_size = (offensive_size < defensive_size) ? offensive_size : defensive_size;
    
    for (int i = 0; i < min_size; i++) {
        int off_idx = offensive_indices[i];
        int def_idx = defensive_indices[i];
        double off_fitness = opt->population[off_idx].fitness;
        double def_fitness = opt->population[def_idx].fitness;
        
        if (off_fitness < def_fitness * WIN_THRESHOLD) {
            winners[(*winner_count)++] = off_idx;
        } else if (def_fitness < off_fitness * WIN_THRESHOLD) {
            winners[(*winner_count)++] = def_idx;
        }
    }
}

// Update Positions with optimized memory access
void update_positions_sgo(Optimizer *opt, int *winners, int winner_count) {
    double r3[opt->dim];
    
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        int is_winner = 0;
        for (int w = 0; w < winner_count; w++) {
            if (winners[w] == i) {
                is_winner = 1;
                break;
            }
        }
        
        if (is_winner && opt->best_solution.fitness != INFINITY) {
            double r2 = rand_double_sgo(0.0, 1.0);
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += ATTACK_RATE * r2 * (opt->best_solution.position[j] - pos[j]);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                r3[j] = rand_double_sgo(0.0, 1.0);
                pos[j] += FIGHT_INTENSITY * r3[j] * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }
        
        // Inline bounds checking
        for (int j = 0; j < opt->dim; j++) {
            if (pos[j] < opt->bounds[2 * j]) {
                pos[j] = opt->bounds[2 * j];
            } else if (pos[j] > opt->bounds[2 * j + 1]) {
                pos[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Main Optimization Function with minimized allocations
void SGO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize players
    initialize_players(opt);
    
    // Allocate memory once
    int offensive_indices[opt->population_size];
    int defensive_indices[opt->population_size];
    int winners[opt->population_size];
    double temp_offensive_pos[opt->dim];
    double temp_defensive_pos[opt->dim];
    int offensive_size, winner_count;
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness with single pass
        double min_fitness = INFINITY;
        int min_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }
        
        // Update best solution
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            memcpy(opt->best_solution.position, opt->population[min_idx].position, 
                   opt->dim * sizeof(double));
        }
        
        // Divide players into teams
        divide_teams(opt, offensive_indices, defensive_indices, &offensive_size);
        
        // Simulate fights
        int fight_count = (offensive_size < (opt->population_size - offensive_size)) 
                         ? offensive_size : (opt->population_size - offensive_size);
        for (int i = 0; i < fight_count; i++) {
            simulate_fight(opt, offensive_indices[i], defensive_indices[i], 
                          temp_offensive_pos, temp_defensive_pos);
        }
        
        // Determine winners and update positions
        determine_winners(opt, offensive_indices, offensive_size, 
                         defensive_indices, opt->population_size - offensive_size, 
                         winners, &winner_count);
        update_positions_sgo(opt, winners, winner_count);
        
        // Optional: Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
