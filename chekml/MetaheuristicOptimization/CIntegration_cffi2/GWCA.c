#include "GWCA.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>  // For parallelization

// GWCA-specific constants
#define G 9.8      // Gravitational constant (m/s^2)
#define M 3        // Some constant for the method
#define E 0.1      // Some constant for the method
#define P 9        // Some constant for the method
#define Q 6        // Some constant for the method
#define CMAX 20    // Max value for constant C
#define CMIN 10    // Min value for constant C

// Compare two solutions based on their fitness values (ascending order)
int compare_fitness(const void *a, const void *b) {
    Solution *sol_a = (Solution*)a;
    Solution *sol_b = (Solution*)b;

    if (sol_a->fitness < sol_b->fitness) return -1;
    if (sol_a->fitness > sol_b->fitness) return 1;
    return 0; // Equal fitness values
}


void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    // Best solution variables
    double best_overall = INFINITY;
    Solution Worker1, Worker2, Worker3;

    // Sort the initial population based on fitness
    qsort(opt->population, opt->population_size, sizeof(Solution), compare_fitness);
    Worker1 = opt->population[0];
    Worker2 = opt->population[1];
    Worker3 = opt->population[2];

    // Initialize fitness for the best overall solution
    best_overall = Worker1.fitness;

    int LNP = (int)ceil(opt->population_size * E);  // Limit of Nearest Population

    // Main GWCA optimization loop
    for (int t = 1; t <= opt->max_iter; t++) {
        // Adjust constant C over iterations
        double C = CMAX - ((CMAX - CMIN) * t / opt->max_iter);

        // Parallel processing using OpenMP
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            double r1 = ((double)rand()) / RAND_MAX;
            double r2 = ((double)rand()) / RAND_MAX;
            double* position = opt->population[i].position;

            // Adjust the position for the first LNP members (local neighborhood)
            if (i < LNP) {
                double F = (M * G * r1) / (P * Q * (1 + t));
                for (int d = 0; d < opt->dim; d++) {
                    position[d] = position[d] + F * (rand() % 2 ? 1 : -1) * C;
                }
            } else {
                // Influence of the three best workers on other members
                for (int d = 0; d < opt->dim; d++) {
                    position[d] = position[d] + r2 * (Worker1.position[d] + Worker2.position[d] + Worker3.position[d]) / 3 * C;
                }
            }

            // Enforce boundary constraints
            enforce_bound_constraints(opt);

            // Calculate fitness for the new position
            opt->population[i].fitness = objective_function(position);

            // Update the best solution if necessary
            if (opt->population[i].fitness < Worker1.fitness) {
                Worker3 = Worker2;
                Worker3.fitness = Worker2.fitness;
                Worker2 = Worker1;
                Worker2.fitness = Worker1.fitness;
                Worker1 = opt->population[i];
            }
        }

        // Track best overall solution
        if (Worker1.fitness < best_overall) {
            best_overall = Worker1.fitness;
            memcpy(opt->best_solution.position, Worker1.position, opt->dim * sizeof(double));
            opt->best_solution.fitness = Worker1.fitness;
        }

        // Optionally print the progress
        printf("Iteration %d: Best Fitness = %f\n", t, Worker1.fitness);
    }
}

