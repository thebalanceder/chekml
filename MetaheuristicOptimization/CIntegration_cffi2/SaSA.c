#include "SaSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize salp population
void sasa_initialize_population(Optimizer *opt) {
    #pragma omp parallel for num_threads(SASA_NUM_THREADS)
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update salp positions (combines leader and follower updates)
void sasa_update_positions(Optimizer *opt, double c1) {
    #pragma omp parallel for num_threads(SASA_NUM_THREADS)
    for (int i = 0; i < opt->population_size; i++) {
        if (i < opt->population_size / 2) {
            // Leader salp update (Eq. 3.1)
            for (int j = 0; j < opt->dim; j++) {
                double lb = opt->bounds[2 * j];
                double ub = opt->bounds[2 * j + 1];
                double c2 = rand_double(0.0, 1.0);
                double c3 = rand_double(0.0, 1.0);
                double new_position;
                if (c3 < 0.5) {
                    new_position = opt->best_solution.position[j] + c1 * ((ub - lb) * c2 + lb);
                } else {
                    new_position = opt->best_solution.position[j] - c1 * ((ub - lb) * c2 + lb);
                }
                opt->population[i].position[j] = new_position;
            }
        } else {
            // Follower salp update (Eq. 3.4)
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = (opt->population[i - 1].position[j] + 
                                                 opt->population[i].position[j]) / 2.0;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all salps
void sasa_evaluate_fitness(Optimizer *opt, double (*objective_function)(double *)) {
    double best_fitness = opt->best_solution.fitness;
    int best_index = -1;
    double *temp_best_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_best_position) {
        fprintf(stderr, "sasa_evaluate_fitness: Memory allocation failed for temp_best_position\n");
        return;
    }

    #pragma omp parallel for num_threads(SASA_NUM_THREADS) reduction(min:best_fitness)
    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = new_fitness;
        if (new_fitness < best_fitness) {
            best_fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < opt->best_solution.fitness) {
                    best_index = i;
                    for (int j = 0; j < opt->dim; j++) {
                        temp_best_position[j] = opt->population[i].position[j];
                    }
                }
            }
        }
    }

    // Update best solution
    if (best_index >= 0) {
        opt->best_solution.fitness = best_fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = temp_best_position[j];
        }
    }

    free(temp_best_position);
}

// Main optimization function
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "SaSA_optimize: Invalid optimizer or objective function\n");
        return;
    }

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Set OpenMP thread count
    omp_set_num_threads(SASA_NUM_THREADS);

    // Allocate temporary array for sorting
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_position) {
        fprintf(stderr, "SaSA_optimize: Memory allocation failed for temp_position\n");
        return;
    }

    // Initialize population
    sasa_initialize_population(opt);

    // Evaluate initial fitness
    sasa_evaluate_fitness(opt, objective_function);

    // Sort population by fitness (bubble sort, value-based)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (opt->population[j].fitness > opt->population[j + 1].fitness) {
                // Swap fitness
                double temp_fitness = opt->population[j].fitness;
                opt->population[j].fitness = opt->population[j + 1].fitness;
                opt->population[j + 1].fitness = temp_fitness;

                // Swap position values (not pointers)
                for (int k = 0; k < opt->dim; k++) {
                    temp_position[k] = opt->population[j].position[k];
                    opt->population[j].position[k] = opt->population[j + 1].position[k];
                    opt->population[j + 1].position[k] = temp_position[k];
                }
            }
        }
    }
    free(temp_position);

    // Precompute constant for c1 calculation
    double c1_scale = SASA_C1_EXPONENT / opt->max_iter;

    // Main optimization loop
    for (int iter = 1; iter <= opt->max_iter; iter++) {
        // Calculate c1 coefficient (Eq. 3.2)
        double c1 = SASA_C1_FACTOR * exp(-pow(c1_scale * iter, 2));

        // Update positions
        sasa_update_positions(opt, c1);

        // Evaluate fitness and update best solution
        sasa_evaluate_fitness(opt, objective_function);

        // Log progress
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    // Final boundary check
    enforce_bound_constraints(opt);
}
