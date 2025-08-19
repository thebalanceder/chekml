#include "CRO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

// Xorshift RNG state
static uint64_t rng_state;

// Initialize RNG with seed
static void init_rng(unsigned int seed) {
    rng_state = seed ? seed : (unsigned int)time(NULL);
    if (rng_state == 0) rng_state = 1;  // Avoid zero state
}

// Fast Xorshift RNG
static uint64_t xorshift64(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

// Macro for random double in [min, max]
#define RAND_DOUBLE(min, max) ((min) + ((max) - (min)) * ((double)xorshift64() / (double)0xFFFFFFFFFFFFFFFFULL))

// Initialize reefs randomly within bounds
void initialize_reefs(Optimizer *opt) {
    if (!opt || !opt->bounds || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer, bounds, or population\n");
        return;
    }

    // Compute solutions per reef
    int solutions_per_reef = opt->population_size / NUM_REEFS;
    if (opt->population_size % NUM_REEFS != 0 || solutions_per_reef <= 0) {
        fprintf(stderr, "Population size %d must be divisible by NUM_REEFS %d\n", opt->population_size, NUM_REEFS);
        return;
    }
    if (solutions_per_reef != POPULATION_SIZE) {
        fprintf(stderr, "Warning: Using %d solutions per reef instead of POPULATION_SIZE %d\n", solutions_per_reef, POPULATION_SIZE);
    }

    // Initialize positions within bounds
    double *positions = opt->population[0].position;  // Contiguous block
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            int idx = i * solutions_per_reef + j;
            double *pos = positions + idx * opt->dim;
            for (int k = 0; k < opt->dim; k++) {
                double lower = opt->bounds[2 * k];
                double upper = opt->bounds[2 * k + 1];
                pos[k] = RAND_DOUBLE(lower, upper);
            }
            opt->population[idx].fitness = INFINITY;
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for modified solutions only
void evaluate_reefs(Optimizer *opt, double (*objective_function)(double *), int *modified, int modified_count) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "Invalid optimizer, population, or objective function\n");
        return;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < modified_count; m++) {
        int idx = modified[m];
        if (idx < 0 || idx >= opt->population_size || !opt->population[idx].position) {
            fprintf(stderr, "Invalid position for solution %d\n", idx);
            continue;
        }
        opt->population[idx].fitness = objective_function(opt->population[idx].position);
    }
}

// Migration phase: selective exchanges
void migration_phase_cfo(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef) {
    if (!opt || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }

    // Perform a fixed number of exchanges (e.g., NUM_REEFS)
    double *positions = opt->population[0].position;
    int total_solutions = opt->population_size;
    for (int k = 0; k < NUM_REEFS; k++) {
        if (*modified_count >= total_solutions) {
            fprintf(stderr, "Warning: modified array full in migration_phase_cfo\n");
            break;
        }
        int i = (int)(RAND_DOUBLE(0, NUM_REEFS));
        int j = (int)(RAND_DOUBLE(0, NUM_REEFS));
        if (i != j) {
            int idx = i * solutions_per_reef + ((int)RAND_DOUBLE(0, solutions_per_reef));
            int idx_replace = j * solutions_per_reef + ((int)RAND_DOUBLE(0, solutions_per_reef));
            if (idx >= total_solutions || idx_replace >= total_solutions) {
                fprintf(stderr, "Invalid index in migration_phase_cfo: idx=%d, idx_replace=%d\n", idx, idx_replace);
                continue;
            }
            double *src = positions + idx * opt->dim;
            double *dst = positions + idx_replace * opt->dim;
            memcpy(dst, src, opt->dim * sizeof(double));
            opt->population[idx_replace].fitness = INFINITY;
            modified[*modified_count] = idx_replace;
            (*modified_count)++;
        }
    }
}

// Local search phase: perturb solutions
void local_search_phase(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef) {
    if (!opt || !opt->population || !opt->population[0].position) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }
    double *positions = opt->population[0].position;
    int total_solutions = opt->population_size;
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            if (*modified_count >= total_solutions) {
                fprintf(stderr, "Warning: modified array full in local_search_phase\n");
                break;
            }
            int idx = i * solutions_per_reef + j;
            if (idx >= total_solutions) {
                fprintf(stderr, "Invalid index in local_search_phase: idx=%d\n", idx);
                continue;
            }
            double *pos = positions + idx * opt->dim;
            for (int k = 0; k < opt->dim; k++) {
                pos[k] += ALPHA * (2.0 * RAND_DOUBLE(0.0, 1.0) - 1.0);
            }
            opt->population[idx].fitness = INFINITY;
            modified[*modified_count] = idx;
            (*modified_count)++;
        }
    }
}

// Main optimization function
void CRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Invalid optimizer or objective function\n");
        return;
    }

    // Initialize RNG
    init_rng((unsigned int)time(NULL));

    // Initialize reefs
    initialize_reefs(opt);
    if (!opt->population || !opt->population[0].position) {
        fprintf(stderr, "Initialization failed\n");
        return;
    }

    int solutions_per_reef = opt->population_size / NUM_REEFS;
    int total_solutions = opt->population_size;

    // Allocate modified solutions array
    int *modified = (int *)malloc(total_solutions * sizeof(int));
    if (!modified) {
        fprintf(stderr, "Failed to allocate modified array\n");
        return;
    }

    // Evaluate initial fitness
    for (int i = 0; i < total_solutions; i++) {
        modified[i] = i;
    }
    evaluate_reefs(opt, objective_function, modified, total_solutions);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        int modified_count = 0;

        // Migration phase
        migration_phase_cfo(opt, modified, &modified_count, solutions_per_reef);

        // Local search phase
        local_search_phase(opt, modified, &modified_count, solutions_per_reef);

        // Evaluate modified solutions
        evaluate_reefs(opt, objective_function, modified, modified_count);

        // Find best solution
        double min_fitness = INFINITY;
        int best_idx = 0;
        double *positions = opt->population[0].position;
        for (int i = 0; i < total_solutions; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                best_idx = i;
            }
        }
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            if (!opt->best_solution.position) {
                fprintf(stderr, "Best solution position is null\n");
                free(modified);
                return;
            }
            memcpy(opt->best_solution.position, positions + best_idx * opt->dim, opt->dim * sizeof(double));
        }

        // Enforce bounds once per iteration
        enforce_bound_constraints(opt);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(modified);
}
