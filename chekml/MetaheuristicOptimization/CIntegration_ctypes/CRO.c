#include "CRO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize reefs randomly within bounds
void initialize_reefs(Optimizer *opt) {
    if (!opt || !opt->bounds || !opt->population) {
        fprintf(stderr, "Invalid optimizer, bounds, or population\n");
        opt->population = NULL;
        return;
    }

    // Compute solutions per reef
    int solutions_per_reef = opt->population_size / NUM_REEFS;
    if (opt->population_size % NUM_REEFS != 0 || solutions_per_reef <= 0) {
        fprintf(stderr, "Population size %d must be divisible by NUM_REEFS %d\n", opt->population_size, NUM_REEFS);
        opt->population = NULL;
        return;
    }
    if (solutions_per_reef != POPULATION_SIZE) {
        fprintf(stderr, "Warning: Using %d solutions per reef instead of POPULATION_SIZE %d\n", solutions_per_reef, POPULATION_SIZE);
    }

    // Initialize positions within bounds
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            int idx = i * solutions_per_reef + j;
            for (int k = 0; k < opt->dim; k++) {
                double lower = opt->bounds[2 * k];
                double upper = opt->bounds[2 * k + 1];
                opt->population[idx].position[k] = lower + (upper - lower) * rand_double(0.0, 1.0);
            }
            opt->population[idx].fitness = INFINITY;
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for each solution in each reef
void evaluate_reefs(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "Invalid optimizer, population, or objective function\n");
        return;
    }
    int solutions_per_reef = opt->population_size / NUM_REEFS;
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            int idx = i * solutions_per_reef + j;
            if (!opt->population[idx].position) {
                fprintf(stderr, "Position for reef %d solution %d is null\n", i, j);
                return;
            }
            opt->population[idx].fitness = objective_function(opt->population[idx].position);
        }
    }
}

// Migration phase: exchange solutions between reefs
void migration_phase_cfo(Optimizer *opt) {
    if (!opt || !opt->population) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }
    int solutions_per_reef = opt->population_size / NUM_REEFS;
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < NUM_REEFS; j++) {
            if (i != j) {
                int idx = i * solutions_per_reef + (rand() % solutions_per_reef);
                int idx_replace = j * solutions_per_reef + (rand() % solutions_per_reef);
                for (int k = 0; k < opt->dim; k++) {
                    opt->population[idx_replace].position[k] = opt->population[idx].position[k];
                }
                opt->population[idx_replace].fitness = INFINITY;  // Mark for re-evaluation
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Local search phase: perturb solutions
void local_search_phase(Optimizer *opt) {
    if (!opt || !opt->population) {
        fprintf(stderr, "Invalid optimizer or population\n");
        return;
    }
    int solutions_per_reef = opt->population_size / NUM_REEFS;
    for (int i = 0; i < NUM_REEFS; i++) {
        for (int j = 0; j < solutions_per_reef; j++) {
            int idx = i * solutions_per_reef + j;
            if (!opt->population[idx].position) {
                fprintf(stderr, "Position for reef %d solution %d is null\n", i, j);
                return;
            }
            for (int k = 0; k < opt->dim; k++) {
                // Uniform perturbation instead of Gaussian for simplicity
                double perturbation = ALPHA * (2.0 * rand_double(0.0, 1.0) - 1.0);
                opt->population[idx].position[k] += perturbation;
            }
            opt->population[idx].fitness = INFINITY;  // Mark for re-evaluation
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void CRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Invalid optimizer or objective function\n");
        return;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize reefs
    initialize_reefs(opt);
    if (!opt->population) {
        fprintf(stderr, "Initialization failed\n");
        return;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        evaluate_reefs(opt, objective_function);

        // Find best solution across all reefs
        double min_fitness = INFINITY;
        int best_reef_idx = 0;
        int best_solution_idx = 0;
        int solutions_per_reef = opt->population_size / NUM_REEFS;
        for (int i = 0; i < NUM_REEFS; i++) {
            for (int j = 0; j < solutions_per_reef; j++) {
                int idx = i * solutions_per_reef + j;
                if (opt->population[idx].fitness < min_fitness) {
                    min_fitness = opt->population[idx].fitness;
                    best_reef_idx = i;
                    best_solution_idx = j;
                }
            }
        }
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            if (!opt->best_solution.position) {
                fprintf(stderr, "Best solution position is null\n");
                return;
            }
            int best_idx = best_reef_idx * solutions_per_reef + best_solution_idx;
            for (int k = 0; k < opt->dim; k++) {
                opt->best_solution.position[k] = opt->population[best_idx].position[k];
            }
        }

        // Migration phase
        migration_phase_cfo(opt);

        // Local search phase
        local_search_phase(opt);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
