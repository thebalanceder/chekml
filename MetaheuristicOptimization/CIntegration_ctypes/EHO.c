/* EHO.c - Implementation file for Elephant Herding Optimization */
#include "EHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void eho_initialize_population(Optimizer *opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) {
        fprintf(stderr, "eho_initialize_population: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "eho_initialize_population: Null position for population[%d]\n", i);
            exit(1);
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
    eho_update_best_solution(opt, objective_function);
}

// Clan Division Phase
void eho_clan_division_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_clan_division_phase: Invalid optimizer\n");
        exit(1);
    }
    
    // Create array of indices
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "eho_clan_division_phase: Memory allocation failed for indices\n");
        exit(1);
    }
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    
    // Sort indices by fitness (ascending)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[indices[j]].fitness > opt->population[indices[j + 1]].fitness) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    // Reorder population based on sorted indices
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        free(indices);
        fprintf(stderr, "eho_clan_division_phase: Memory allocation failed for temp_position\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        if (indices[i] != i) {
            // Swap position and fitness
            for (int j = 0; j < opt->dim; j++) {
                temp_position[j] = opt->population[i].position[j];
                opt->population[i].position[j] = opt->population[indices[i]].position[j];
                opt->population[indices[i]].position[j] = temp_position[j];
            }
            double temp_fitness = opt->population[i].fitness;
            opt->population[i].fitness = opt->population[indices[i]].fitness;
            opt->population[indices[i]].fitness = temp_fitness;
            
            // Update indices to reflect swap
            for (int k = 0; k < opt->population_size; k++) {
                if (indices[k] == i) {
                    indices[k] = indices[i];
                    indices[i] = i;
                    break;
                }
            }
        }
    }
    
    free(temp_position);
    free(indices);
}

// Clan Updating Phase
void eho_clan_updating_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_clan_updating_phase: Invalid optimizer\n");
        exit(1);
    }
    
    int elephants_per_clan = opt->population_size / EHO_NUM_CLANS;
    for (int c = 0; c < EHO_NUM_CLANS; c++) {
        // Compute clan center
        double clan_center[opt->dim];
        for (int j = 0; j < opt->dim; j++) {
            clan_center[j] = 0.0;
            for (int i = 0; i < elephants_per_clan; i++) {
                clan_center[j] += opt->population[c * elephants_per_clan + i].position[j];
            }
            clan_center[j] /= elephants_per_clan;
        }
        
        // Update each elephant in the clan
        for (int i = 0; i < elephants_per_clan; i++) {
            int idx = c * elephants_per_clan + i;
            double *current = opt->population[idx].position;
            double *best = opt->population[c * elephants_per_clan].position; // Best in clan
            double new_pos[opt->dim];
            
            // Clan updating rule
            for (int j = 0; j < opt->dim; j++) {
                new_pos[j] = current[j] + EHO_ALPHA * (best[j] - current[j]) * rand_double(0.0, 1.0);
            }
            
            // If no change, move towards clan center
            int unchanged = 1;
            for (int j = 0; j < opt->dim; j++) {
                if (new_pos[j] != current[j]) {
                    unchanged = 0;
                    break;
                }
            }
            if (unchanged) {
                for (int j = 0; j < opt->dim; j++) {
                    new_pos[j] = EHO_BETA * clan_center[j];
                }
            }
            
            // Clip to bounds and update position
            for (int j = 0; j < opt->dim; j++) {
                if (new_pos[j] < opt->bounds[2 * j]) new_pos[j] = opt->bounds[2 * j];
                if (new_pos[j] > opt->bounds[2 * j + 1]) new_pos[j] = opt->bounds[2 * j + 1];
                opt->population[idx].position[j] = new_pos[j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Separating Phase
void eho_separating_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_separating_phase: Invalid optimizer\n");
        exit(1);
    }
    
    int elephants_per_clan = opt->population_size / EHO_NUM_CLANS;
    for (int c = 0; c < EHO_NUM_CLANS; c++) {
        // Replace worst elephant in each clan
        int worst_idx = c * elephants_per_clan + elephants_per_clan - 1;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[worst_idx].position[j] = opt->bounds[2 * j] + 
                                                    rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[worst_idx].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Elitism Phase
void eho_elitism_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_elitism_phase: Invalid optimizer\n");
        exit(1);
    }
    
    // Store elite solutions
    Solution elite[EHO_KEEP];
    for (int i = 0; i < EHO_KEEP; i++) {
        elite[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!elite[i].position) {
            for (int j = 0; j < i; j++) free(elite[j].position);
            fprintf(stderr, "eho_elitism_phase: Memory allocation failed\n");
            exit(1);
        }
        fprintf(stderr, "eho_elitism_phase: Allocating elite[%d].position at %p\n", i, elite[i].position);
        for (int j = 0; j < opt->dim; j++) {
            elite[i].position[j] = opt->population[i].position[j];
        }
        elite[i].fitness = opt->population[i].fitness;
    }
    
    // Sort population
    eho_clan_division_phase(opt);
    
    // Replace worst with elites
    for (int i = 0; i < EHO_KEEP; i++) {
        fprintf(stderr, "eho_elitism_phase: Copying elite[%d].position to population[%d]\n", 
                i, opt->population_size - 1 - i);
        for (int j = 0; j < opt->dim; j++) {
            opt->population[opt->population_size - 1 - i].position[j] = elite[i].position[j];
        }
        opt->population[opt->population_size - 1 - i].fitness = elite[i].fitness;
    }
    
    // Free elite positions
    for (int i = 0; i < EHO_KEEP; i++) {
        fprintf(stderr, "eho_elitism_phase: Freeing elite[%d].position at %p\n", i, elite[i].position);
        free(elite[i].position);
        elite[i].position = NULL;
    }
}

// Update Best Solution
void eho_update_best_solution(Optimizer *opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) {
        fprintf(stderr, "eho_update_best_solution: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < opt->best_solution.fitness) {
            fprintf(stderr, "eho_update_best_solution: Updating best_solution with population[%d], fitness=%f\n", 
                    i, fitness);
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Main Optimization Function
void EHO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    if (!opt_void || !objective_function) {
        fprintf(stderr, "EHO_optimize: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    Optimizer *opt = (Optimizer *)opt_void;
    
    // Validate population size
    if (opt->population_size % EHO_NUM_CLANS != 0) {
        fprintf(stderr, "EHO_optimize: Population size must be divisible by number of clans\n");
        exit(1);
    }
    
    eho_initialize_population(opt, objective_function);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        eho_clan_division_phase(opt);
        eho_clan_updating_phase(opt);
        eho_separating_phase(opt);
        
        // Evaluate population
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        
        eho_elitism_phase(opt);
        eho_update_best_solution(opt, objective_function);
    }
}
