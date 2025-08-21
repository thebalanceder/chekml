#include "LoSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator
#include <string.h>  // For memcpy
#ifdef _OPENMP
#include <omp.h>     // For parallelization
#endif

// Initialize Population
void LoSA_initialize_population(Optimizer *opt) {
#ifdef LOSA_DEBUG
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "Error: Null pointer in LoSA_initialize_population\n");
        return;
    }
#endif

    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *bounds = opt->bounds;

    #pragma omp parallel for
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) {
            opt->population[i].position[j] = bounds[2 * j] + 
                                            (bounds[2 * j + 1] - bounds[2 * j]) * rand_double_LoSA(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update Positions
void LoSA_update_positions(Optimizer *opt) {
#ifdef LOSA_DEBUG
    if (!opt || !opt->population || !opt->best_solution.position) {
        fprintf(stderr, "Error: Null pointer in LoSA_update_positions\n");
        return;
    }
#endif

    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *best_pos = opt->best_solution.position;

    // Precompute random numbers for all individuals
    double *rand_vals = (double *)malloc(pop_size * dim * sizeof(double));
#ifdef LOSA_DEBUG
    if (!rand_vals) {
        fprintf(stderr, "Error: Memory allocation failed in LoSA_update_positions\n");
        return;
    }
#endif
    for (int i = 0; i < pop_size * dim; i++) {
        rand_vals[i] = rand_double_LoSA(0.0, 1.0);
    }

    #pragma omp parallel for
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double direction = best_pos[j] - pos[j];
            pos[j] += LOSA_STEP_SIZE * direction * rand_vals[i * dim + j];
        }
    }

    free(rand_vals);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
#ifdef LOSA_DEBUG
    if (!opt || !objective_function || !opt->population || !opt->bounds || 
        opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid parameters in LoSA_optimize\n");
        return;
    }
#endif

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Cache frequently accessed values
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int max_iter = opt->max_iter;
    double *best_pos = opt->best_solution.position;
    double *best_fitness = &opt->best_solution.fitness;

    // Initialize population
    LoSA_initialize_population(opt);
    *best_fitness = INFINITY;

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Evaluate fitness for each individual (parallelized)
        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < *best_fitness) {
                    *best_fitness = new_fitness;
                    memcpy(best_pos, opt->population[i].position, dim * sizeof(double));
                }
            }
        }

        // Update positions
        LoSA_update_positions(opt);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, *best_fitness);
    }
}
