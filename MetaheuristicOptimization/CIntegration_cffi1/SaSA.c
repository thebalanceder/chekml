#include "SaSA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Inline random double generator
#define RAND_DOUBLE(min, max) (min + (max - min) * ((double)rand() / RAND_MAX))

// Main optimization function
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || opt->dim != SASA_DIM || 
        opt->population_size != SASA_POPULATION_SIZE || opt->max_iter != SASA_MAX_ITERATIONS) {
        fprintf(stderr, "SaSA_optimize: Invalid optimizer or parameters\n");
        return;
    }

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Set OpenMP thread count
    omp_set_num_threads(SASA_NUM_THREADS);

    // Initialize population
    #pragma omp parallel for num_threads(SASA_NUM_THREADS)
    for (int i = 0; i < SASA_POPULATION_SIZE; i++) {
        opt->population[i].position[0] = RAND_DOUBLE(opt->bounds[0], opt->bounds[1]);
        opt->population[i].position[1] = RAND_DOUBLE(opt->bounds[2], opt->bounds[3]);
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);

    // Precompute c1 scale
    double c1_scale = SASA_C1_EXPONENT / SASA_MAX_ITERATIONS;

    // Main optimization loop
    for (int iter = 1; iter <= SASA_MAX_ITERATIONS; iter++) {
        // Calculate c1 coefficient (Eq. 3.2)
        double c1 = SASA_C1_FACTOR * exp(-pow(c1_scale * iter, 2));

        // Update positions and evaluate fitness in one parallel loop
        double best_fitness = opt->best_solution.fitness;
        int best_index = -1;
        double temp_best_pos[SASA_DIM];

        #pragma omp parallel for num_threads(SASA_NUM_THREADS) reduction(min:best_fitness)
        for (int i = 0; i < SASA_POPULATION_SIZE; i++) {
            // Update positions
            if (i < SASA_POPULATION_SIZE / 2) {
                // Leader salp update (Eq. 3.1)
                double c2_0 = RAND_DOUBLE(0.0, 1.0);
                double c2_1 = RAND_DOUBLE(0.0, 1.0);
                double c3_0 = RAND_DOUBLE(0.0, 1.0);
                double c3_1 = RAND_DOUBLE(0.0, 1.0);

                // Unroll for dim=2
                if (c3_0 < 0.5) {
                    opt->population[i].position[0] = opt->best_solution.position[0] + 
                        c1 * ((opt->bounds[1] - opt->bounds[0]) * c2_0 + opt->bounds[0]);
                } else {
                    opt->population[i].position[0] = opt->best_solution.position[0] - 
                        c1 * ((opt->bounds[1] - opt->bounds[0]) * c2_0 + opt->bounds[0]);
                }
                if (c3_1 < 0.5) {
                    opt->population[i].position[1] = opt->best_solution.position[1] + 
                        c1 * ((opt->bounds[3] - opt->bounds[2]) * c2_1 + opt->bounds[2]);
                } else {
                    opt->population[i].position[1] = opt->best_solution.position[1] - 
                        c1 * ((opt->bounds[3] - opt->bounds[2]) * c2_1 + opt->bounds[2]);
                }
            } else {
                // Follower salp update (Eq. 3.4)
                opt->population[i].position[0] = (opt->population[i - 1].position[0] + 
                                                 opt->population[i].position[0]) / 2.0;
                opt->population[i].position[1] = (opt->population[i - 1].position[1] + 
                                                 opt->population[i].position[1]) / 2.0;
            }

            // Inline boundary check to reduce calls
            if (opt->population[i].position[0] < opt->bounds[0]) opt->population[i].position[0] = opt->bounds[0];
            if (opt->population[i].position[0] > opt->bounds[1]) opt->population[i].position[0] = opt->bounds[1];
            if (opt->population[i].position[1] < opt->bounds[2]) opt->population[i].position[1] = opt->bounds[2];
            if (opt->population[i].position[1] > opt->bounds[3]) opt->population[i].position[1] = opt->bounds[3];

            // Evaluate fitness
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;

            // Update best solution
            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                #pragma omp critical
                {
                    if (new_fitness < opt->best_solution.fitness) {
                        best_index = i;
                        temp_best_pos[0] = opt->population[i].position[0];
                        temp_best_pos[1] = opt->population[i].position[1];
                    }
                }
            }
        }

        // Update best solution
        if (best_index >= 0) {
            opt->best_solution.fitness = best_fitness;
            opt->best_solution.position[0] = temp_best_pos[0];
            opt->best_solution.position[1] = temp_best_pos[1];
        }

        // Log progress
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    // Final boundary check
    enforce_bound_constraints(opt);
}
