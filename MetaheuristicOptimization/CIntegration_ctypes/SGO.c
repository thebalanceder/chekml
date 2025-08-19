/* SGO.c - Implementation file for Squid Game Optimization */
#include "SGO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if you want to seed the random generator
#include <string.h>  // For memcpy()

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Players (Population)
void initialize_players(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;  // Initial fitness
    }
    enforce_bound_constraints(opt);
}

// Divide Players into Offensive and Defensive Teams
void divide_teams(Optimizer *opt, int *offensive_indices, int *defensive_indices, int *offensive_size) {
    *offensive_size = (int)(opt->population_size * ATTACK_RATE);
    int indices[opt->population_size];
    
    // Initialize indices
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Shuffle indices (Fisher-Yates shuffle)
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Split into offensive and defensive teams
    for (int i = 0; i < *offensive_size; i++) {
        offensive_indices[i] = indices[i];
    }
    for (int i = *offensive_size; i < opt->population_size; i++) {
        defensive_indices[i - *offensive_size] = indices[i];
    }
}

// Simulate Fight between Offensive and Defensive Players
void simulate_fight(Optimizer *opt, int offensive_idx, int defensive_idx) {
    double r1 = rand_double(0.0, 1.0);
    double movement[opt->dim], resistance[opt->dim];
    
    // Calculate movement vector based on fight intensity
    for (int j = 0; j < opt->dim; j++) {
        movement[j] = FIGHT_INTENSITY * (opt->population[offensive_idx].position[j] - 
                                         opt->population[defensive_idx].position[j]) * r1;
    }
    
    // Update offensive player position
    for (int j = 0; j < opt->dim; j++) {
        opt->population[offensive_idx].position[j] += movement[j];
        if (opt->population[offensive_idx].position[j] < opt->bounds[2 * j]) {
            opt->population[offensive_idx].position[j] = opt->bounds[2 * j];
        } else if (opt->population[offensive_idx].position[j] > opt->bounds[2 * j + 1]) {
            opt->population[offensive_idx].position[j] = opt->bounds[2 * j + 1];
        }
    }
    
    // Update defensive player position with resistance
    for (int j = 0; j < opt->dim; j++) {
        resistance[j] = DEFENSE_STRENGTH * (opt->population[defensive_idx].position[j] - 
                                            opt->population[offensive_idx].position[j]) * (1.0 - r1);
        opt->population[defensive_idx].position[j] += resistance[j];
        if (opt->population[defensive_idx].position[j] < opt->bounds[2 * j]) {
            opt->population[defensive_idx].position[j] = opt->bounds[2 * j];
        } else if (opt->population[defensive_idx].position[j] > opt->bounds[2 * j + 1]) {
            opt->population[defensive_idx].position[j] = opt->bounds[2 * j + 1];
        }
    }
}

// Determine Winners based on Fitness
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
            winners[*winner_count] = off_idx;
            (*winner_count)++;
        } else if (def_fitness < off_fitness * WIN_THRESHOLD) {
            winners[*winner_count] = def_idx;
            (*winner_count)++;
        }
    }
}

// Update Positions based on Winning States
void update_positions_sgo(Optimizer *opt, int *winners, int winner_count) {
    double r3[opt->dim];
    
    for (int i = 0; i < opt->population_size; i++) {
        int is_winner = 0;
        for (int w = 0; w < winner_count; w++) {
            if (winners[w] == i) {
                is_winner = 1;
                break;
            }
        }
        
        if (is_winner && opt->best_solution.fitness != INFINITY) {
            // Winners move towards best solution
            double r2 = rand_double(0.0, 1.0);
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] += ATTACK_RATE * r2 * 
                                                 (opt->best_solution.position[j] - opt->population[i].position[j]);
            }
        } else {
            // Losers undergo random perturbation
            for (int j = 0; j < opt->dim; j++) {
                r3[j] = rand_double(0.0, 1.0);
                opt->population[i].position[j] += FIGHT_INTENSITY * r3[j] * 
                                                 (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }
        
        // Clip to bounds
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Main Optimization Function
void SGO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize players
    initialize_players(opt);
    
    // Allocate memory for team indices and winners
    int offensive_indices[opt->population_size];
    int defensive_indices[opt->population_size];
    int winners[opt->population_size];
    int offensive_size, winner_count;
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        
        // Divide players into offensive and defensive teams
        divide_teams(opt, offensive_indices, defensive_indices, &offensive_size);
        
        // Simulate fights between offensive and defensive players
        for (int i = 0; i < offensive_size && i < (opt->population_size - offensive_size); i++) {
            simulate_fight(opt, offensive_indices[i], defensive_indices[i]);
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
