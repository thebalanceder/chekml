#include "CMOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Fast Xorshift random number generator
static unsigned int xorshift_state = 1;
static inline unsigned int xorshift_rand() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static inline double fast_rand_double(double min, double max) {
    return min + (max - min) * ((double)xorshift_rand() / (double)0xFFFFFFFF);
}

// Helper function to find the index of the closest solution to solution i
static int find_closest_solution(Optimizer *opt, int i) {
    double min_distance = INFINITY;
    int closest_idx = i;
    double *pos_i = opt->population[i].position;
    for (int k = 0; k < opt->population_size; k++) {
        if (k == i) continue;
        double distance = 0.0;
        double *pos_k = opt->population[k].position;
        for (int j = 0; j < opt->dim; j++) {
            distance += fabs(pos_k[j] - pos_i[j]);
        }
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = k;
        }
    }
    return closest_idx;
}

// Combined CMOA Phases
static void update_solution(Optimizer *opt, int i, int t, double *randoms, int *random_indices, double cos_t) {
    double *pos = opt->population[i].position;
    int dim = opt->dim;
    double *bounds = opt->bounds;
    double *best_pos = opt->best_solution.position;
    int has_best = (opt->best_solution.fitness != INFINITY);

    // Genetic Recombination
    if (randoms[0] < CMOA_CROSSOVER_RATE) {
        int closest_idx = find_closest_solution(opt, i);
        double *closest_pos = opt->population[closest_idx].position;
        for (int j = 0; j < dim; j++) {
            pos[j] += randoms[1] * (closest_pos[j] - pos[j]);
        }
    }

    // Cross-Activation
    if (has_best) {
        for (int j = 0; j < dim; j++) {
            pos[j] += randoms[2] * (best_pos[j] - pos[j]);
        }
    }

    // Incremental Reactivation
    if (has_best) {
        for (int j = 0; j < dim; j++) {
            pos[j] += cos_t * randoms[3] * (best_pos[j] - pos[j]);
        }
    }

    // Non-Genetic Mutation
    if (randoms[4] < CMOA_MUTATION_RATE) {
        for (int j = 0; j < dim; j++) {
            double range = bounds[2 * j + 1] - bounds[2 * j];
            pos[j] += randoms[5] * CMOA_MUTATION_SCALE * range;
        }
    }

    // Genotypic Mixing
    int idx1 = random_indices[0];
    int idx2 = random_indices[1];
    double *pos1 = opt->population[idx1].position;
    double *pos2 = opt->population[idx2].position;
    for (int j = 0; j < dim; j++) {
        pos[j] += randoms[6] * (pos1[j] - pos2[j]);
    }
}

// Main Optimization Function
void CMOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize random seed for Xorshift
    xorshift_state = (unsigned int)time(NULL);

    // Precompute cosine values for incremental reactivation
    double *cos_table = (double *)malloc(opt->max_iter * sizeof(double));
    for (int t = 0; t < opt->max_iter; t++) {
        cos_table[t] = cos(M_PI * ((double)t / (double)opt->max_iter));
    }

    // Allocate buffers for random numbers and indices
    double *randoms = (double *)malloc(7 * sizeof(double));  // For each phase
    int *random_indices = (int *)malloc(2 * sizeof(int));    // For genotypic mixing

    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update all solutions
        for (int i = 0; i < opt->population_size; i++) {
            // Generate random numbers for this solution
            for (int r = 0; r < 7; r++) {
                randoms[r] = fast_rand_double(r < 5 ? 0.0 : -1.0, 1.0);  // [-1, 1] for mutation
            }
            // Generate random indices for genotypic mixing
            random_indices[0] = xorshift_rand() % opt->population_size;
            do {
                random_indices[1] = xorshift_rand() % opt->population_size;
            } while (random_indices[1] == random_indices[0]);

            // Update solution with all phases
            update_solution(opt, i, iter, randoms, random_indices, cos_table[iter]);
        }

        // Enforce bounds after all updates
        enforce_bound_constraints(opt);

        // Evaluate population and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(cos_table);
    free(randoms);
    free(random_indices);
}
