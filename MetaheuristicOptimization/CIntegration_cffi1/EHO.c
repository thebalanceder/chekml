/* EHO.c - Extremely Optimized Elephant Herding Optimization for CPU */
#include <stdint.h>
#include "EHO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// XORShift128+ RNG for fast, high-quality random numbers
static uint64_t rng_state[2] = {123456789, 987654321};

static inline double fast_rand_double(double min, double max) {
    uint64_t x = rng_state[0];
    uint64_t y = rng_state[1];
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    rng_state[0] = y;
    rng_state[1] = x;
    double r = (double)(x >> 12) / (1ULL << 52);
    return min + (max - min) * r;
}

// In-place quicksort with insertion sort for small arrays
static void swap_positions(Optimizer *opt, int i, int j, double *temp_position) {
    memcpy(temp_position, opt->population[i].position, opt->dim * sizeof(double));
    memcpy(opt->population[i].position, opt->population[j].position, opt->dim * sizeof(double));
    memcpy(opt->population[j].position, temp_position, opt->dim * sizeof(double));
    double temp_fitness = opt->population[i].fitness;
    opt->population[i].fitness = opt->population[j].fitness;
    opt->population[j].fitness = temp_fitness;
}

static void insertion_sort(Optimizer *opt, int low, int high, double *temp_position) {
    for (int i = low + 1; i <= high; i++) {
        double key = opt->population[i].fitness;
        int j = i - 1;
        while (j >= low && opt->population[j].fitness > key) {
            swap_positions(opt, j, j + 1, temp_position);
            j--;
        }
    }
}

static int partition(Optimizer *opt, int low, int high, double *temp_position) {
    double pivot = opt->population[high].fitness;
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (opt->population[j].fitness <= pivot) {
            i++;
            swap_positions(opt, i, j, temp_position);
        }
    }
    swap_positions(opt, i + 1, high, temp_position);
    return i + 1;
}

static void quicksort(Optimizer *opt, int low, int high, double *temp_position) {
    if (high - low < 10) {
        insertion_sort(opt, low, high, temp_position);
        return;
    }
    if (low < high) {
        int pi = partition(opt, low, high, temp_position);
        quicksort(opt, low, pi - 1, temp_position);
        quicksort(opt, pi + 1, high, temp_position);
    }
}

// Initialize Population
void eho_initialize_population(Optimizer *opt, ObjectiveFunction objective_function, double *temp_position) {
    if (!opt || !objective_function) {
        fprintf(stderr, "eho_initialize_population: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double min_b = opt->bounds[2 * j], max_b = opt->bounds[2 * j + 1];
            pos[j] = min_b + fast_rand_double(0.0, 1.0) * (max_b - min_b);
            if (pos[j] < min_b) pos[j] = min_b;
            if (pos[j] > max_b) pos[j] = max_b;
        }
        opt->population[i].fitness = objective_function(pos);
    }
}

// Clan Division Phase (In-place Quicksort)
void eho_clan_division_phase(Optimizer *opt, double *temp_position) {
    if (!opt) {
        fprintf(stderr, "eho_clan_division_phase: Invalid optimizer\n");
        exit(1);
    }
    quicksort(opt, 0, opt->population_size - 1, temp_position);
}

// Clan Updating Phase (Vectorized and Parallelized)
void eho_clan_updating_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_clan_updating_phase: Invalid optimizer\n");
        exit(1);
    }
    
    int elephants_per_clan = opt->population_size / EHO_NUM_CLANS;
    double clan_centers[EHO_NUM_CLANS][opt->dim];
    
    // Compute clan centers
    #pragma omp parallel for
    for (int c = 0; c < EHO_NUM_CLANS; c++) {
        for (int j = 0; j < opt->dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < elephants_per_clan; i++) {
                sum += opt->population[c * elephants_per_clan + i].position[j];
            }
            clan_centers[c][j] = sum / elephants_per_clan;
        }
    }
    
    // Update elephants
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < EHO_NUM_CLANS; c++) {
        double *best = opt->population[c * elephants_per_clan].position;
        double *center = clan_centers[c];
        for (int i = 0; i < elephants_per_clan; i++) {
            int idx = c * elephants_per_clan + i;
            double *current = opt->population[idx].position;
            double new_pos[opt->dim];
            
            // Unrolled update for small dim (e.g., dim=2)
            if (opt->dim == 2) {
                double r0 = fast_rand_double(0.0, 1.0);
                double r1 = fast_rand_double(0.0, 1.0);
                new_pos[0] = current[0] + EHO_ALPHA * (best[0] - current[0]) * r0;
                new_pos[1] = current[1] + EHO_ALPHA * (best[1] - current[1]) * r1;
                if (new_pos[0] == current[0]) new_pos[0] = EHO_BETA * center[0];
                if (new_pos[1] == current[1]) new_pos[1] = EHO_BETA * center[1];
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    new_pos[j] = current[j] + EHO_ALPHA * (best[j] - current[j]) * fast_rand_double(0.0, 1.0);
                    if (new_pos[j] == current[j]) new_pos[j] = EHO_BETA * center[j];
                }
            }
            
            // Inline bounds check
            for (int j = 0; j < opt->dim; j++) {
                double min_b = opt->bounds[2 * j], max_b = opt->bounds[2 * j + 1];
                new_pos[j] = new_pos[j] < min_b ? min_b : new_pos[j] > max_b ? max_b : new_pos[j];
                current[j] = new_pos[j];
            }
        }
    }
}

// Separating Phase
void eho_separating_phase(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "eho_separating_phase: Invalid optimizer\n");
        exit(1);
    }
    
    int elephants_per_clan = opt->population_size / EHO_NUM_CLANS;
    #pragma omp parallel for
    for (int c = 0; c < EHO_NUM_CLANS; c++) {
        int idx = c * elephants_per_clan + elephants_per_clan - 1;
        double *pos = opt->population[idx].position;
        for (int j = 0; j < opt->dim; j++) {
            double min_b = opt->bounds[2 * j], max_b = opt->bounds[2 * j + 1];
            pos[j] = min_b + fast_rand_double(0.0, 1.0) * (max_b - min_b);
            if (pos[j] < min_b) pos[j] = min_b;
            if (pos[j] > max_b) pos[j] = max_b;
        }
        opt->population[idx].fitness = INFINITY;
    }
}

// Elitism Phase
void eho_elitism_phase(Optimizer *opt, Solution *elite, double *temp_position) {
    if (!opt) {
        fprintf(stderr, "eho_elitism_phase: Invalid optimizer\n");
        exit(1);
    }
    
    // Store elite solutions
    for (int i = 0; i < EHO_KEEP; i++) {
        memcpy(elite[i].position, opt->population[i].position, opt->dim * sizeof(double));
        elite[i].fitness = opt->population[i].fitness;
    }
    
    // Sort population
    eho_clan_division_phase(opt, temp_position);
    
    // Replace worst with elites
    for (int i = 0; i < EHO_KEEP; i++) {
        int idx = opt->population_size - 1 - i;
        memcpy(opt->population[idx].position, elite[i].position, opt->dim * sizeof(double));
        opt->population[idx].fitness = elite[i].fitness;
    }
}

// Update Best Solution
void eho_update_best_solution(Optimizer *opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) {
        fprintf(stderr, "eho_update_best_solution: Invalid optimizer or objective function\n");
        exit(1);
    }
    
    double best_fitness = opt->best_solution.fitness;
    int best_idx = -1;
    
    #pragma omp parallel for reduction(min:best_fitness)
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            #pragma omp critical
            if (fitness < opt->best_solution.fitness) {
                best_idx = i;
            }
        }
    }
    
    if (best_idx >= 0) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[best_idx].position, opt->dim * sizeof(double));
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
    
    // Preallocate buffers
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    Solution *elite = (Solution *)malloc(EHO_KEEP * sizeof(Solution));
    if (!temp_position || !elite) {
        free(temp_position);
        free(elite);
        fprintf(stderr, "EHO_optimize: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < EHO_KEEP; i++) {
        elite[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!elite[i].position) {
            for (int j = 0; j < i; j++) free(elite[j].position);
            free(temp_position);
            free(elite);
            fprintf(stderr, "EHO_optimize: Memory allocation failed for elite[%d]\n", i);
            exit(1);
        }
    }
    
    // Initialize population
    eho_initialize_population(opt, objective_function, temp_position);
    eho_update_best_solution(opt, objective_function);
    
    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        eho_clan_division_phase(opt, temp_position);
        eho_clan_updating_phase(opt);
        eho_separating_phase(opt);
        
        // Evaluate population
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        
        eho_elitism_phase(opt, elite, temp_position);
        eho_update_best_solution(opt, objective_function);
    }
    
    // Cleanup
    for (int i = 0; i < EHO_KEEP; i++) {
        free(elite[i].position);
    }
    free(elite);
    free(temp_position);
}
