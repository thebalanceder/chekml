/* EHO.c - Optimized Implementation for Elephant Herding Optimization */
#include "EHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Linear Congruential Generator for faster random numbers
static unsigned long lcg_seed = 123456789;
static inline double fast_rand_double(double min, double max) {
    lcg_seed = (6364136223846793005ULL * lcg_seed + 1442695040888963407ULL);
    double r = (double)(lcg_seed >> 32) / 0xFFFFFFFF;
    return min + (max - min) * r;
}

// Quicksort helper functions
static void swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

static int partition(int *indices, double *fitnesses, int low, int high) {
    double pivot = fitnesses[indices[high]];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (fitnesses[indices[j]] <= pivot) {
            i++;
            swap_int(&indices[i], &indices[j]);
        }
    }
    swap_int(&indices[i + 1], &indices[high]);
    return i + 1;
}

static void quicksort(int *indices, double *fitnesses, int low, int high) {
    if (low < high) {
        int pi = partition(indices, fitnesses, low, high);
        quicksort(indices, fitnesses, low, pi - 1);
        quicksort(indices, fitnesses, pi + 1, high);
    }
}

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
                                            fast_rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    enforce_bound_constraints(opt);
    eho_update_best_solution(opt, objective_function);
}

// Clan Division Phase (Optimized with Quicksort)
void eho_clan_division_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_clan_division_phase: Invalid optimizer\n");
        exit(1);
    }
    
    // Allocate indices and fitness array
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    double *fitnesses = (double *)malloc(opt->population_size * sizeof(double));
    if (!indices || !fitnesses) {
        free(indices);
        free(fitnesses);
        fprintf(stderr, "eho_clan_division_phase: Memory allocation failed\n");
        exit(1);
    }
    
    // Initialize indices and copy fitnesses
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
        fitnesses[i] = opt->population[i].fitness;
    }
    
    // Sort indices using quicksort
    quicksort(indices, fitnesses, 0, opt->population_size - 1);
    
    // Reorder population based on sorted indices
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        free(indices);
        free(fitnesses);
        fprintf(stderr, "eho_clan_division_phase: Memory allocation failed for temp_position\n");
        exit(1);
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        if (indices[i] != i) {
            // Swap position and fitness
            memcpy(temp_position, opt->population[i].position, opt->dim * sizeof(double));
            memcpy(opt->population[i].position, opt->population[indices[i]].position, opt->dim * sizeof(double));
            memcpy(opt->population[indices[i]].position, temp_position, opt->dim * sizeof(double));
            
            double temp_fitness = opt->population[i].fitness;
            opt->population[i].fitness = opt->population[indices[i]].fitness;
            opt->population[indices[i]].fitness = temp_fitness;
            
            // Update indices
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
    free(fitnesses);
    free(indices);
}

// Clan Updating Phase (Parallelized)
void eho_clan_updating_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_clan_updating_phase: Invalid optimizer\n");
        exit(1);
    }
    
    int elephants_per_clan = opt->population_size / EHO_NUM_CLANS;
    
    #pragma omp parallel for schedule(dynamic)
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
            double *best = opt->population[c * elephants_per_clan].position;
            double new_pos[opt->dim];
            
            // Clan updating rule (always applies to ensure movement)
            for (int j = 0; j < opt->dim; j++) {
                new_pos[j] = current[j] + EHO_ALPHA * (best[j] - current[j]) * fast_rand_double(0.0, 1.0);
                if (new_pos[j] == current[j]) { // Rare case: ensure movement
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
        int worst_idx = c * elephants_per_clan + elephants_per_clan - 1;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[worst_idx].position[j] = opt->bounds[2 * j] + 
                                                    fast_rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[worst_idx].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Elitism Phase (Reused Memory)
void eho_elitism_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_elitism_phase: Invalid optimizer\n");
        exit(1);
    }
    
    // Static elite array (allocated once)
    static Solution elite[EHO_KEEP];
    static int initialized = 0;
    if (!initialized) {
        for (int i = 0; i < EHO_KEEP; i++) {
            elite[i].position = (double *)malloc(opt->dim * sizeof(double));
            if (!elite[i].position) {
                for (int j = 0; j < i; j++) free(elite[j].position);
                fprintf(stderr, "eho_elitism_phase: Memory allocation failed\n");
                exit(1);
            }
            fprintf(stderr, "eho_elitism_phase: Allocating elite[%d].position at %p\n", i, elite[i].position);
        }
        initialized = 1;
    }
    
    // Store elite solutions
    for (int i = 0; i < EHO_KEEP; i++) {
        memcpy(elite[i].position, opt->population[i].position, opt->dim * sizeof(double));
        elite[i].fitness = opt->population[i].fitness;
    }
    
    // Sort population
    eho_clan_division_phase(opt);
    
    // Replace worst with elites
    for (int i = 0; i < EHO_KEEP; i++) {
        fprintf(stderr, "eho_elitism_phase: Copying elite[%d].position to population[%d]\n", 
                i, opt->population_size - 1 - i);
        memcpy(opt->population[opt->population_size - 1 - i].position, elite[i].position, opt->dim * sizeof(double));
        opt->population[opt->population_size - 1 - i].fitness = elite[i].fitness;
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
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
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
    
    if (opt->population_size % EHO_NUM_CLANS != 0) {
        fprintf(stderr, "EHO_optimize: Population size must be divisible by number of EHO_NUM_CLANS\n");
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
