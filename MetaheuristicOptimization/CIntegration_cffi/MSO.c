#include "MSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Initialize Xorshift RNG
void mso_xorshift_init(MSO_XorshiftState *state, unsigned long seed) {
    state->x = seed | 1; // Ensure non-zero
    state->y = 1812433253UL * state->x + 1;
    state->z = 1812433253UL * state->y + 1;
    state->w = 1812433253UL * state->z + 1;
}

// Update positions (optimized with loop unrolling and local caching)
void mso_update_positions(Optimizer *opt, int iter, MSO_XorshiftState *rng_state) {
    double p_explore = fmax(MSO_MAX_P_EXPLORE * exp(-0.1 * iter), MSO_MIN_P_EXPLORE);
    double *best_pos = opt->best_solution.position;
    double *bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    // Precompute frequently used values
    double p_explore_threshold = p_explore;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;

        // Unroll loop for small dimensions (common case: dim=2)
        if (dim == 2) {
            double rand_val = mso_xorshift_double(rng_state, 0.0, 1.0);
            if (rand_val < p_explore_threshold) { // Exploration
                double r = mso_xorshift_double(rng_state, 0.0, 1.0);
                if (r < 0.5) {
                    pos[0] = best_pos[0] + mso_xorshift_double(rng_state, 0.0, 1.0) * (bounds[1] - best_pos[0]);
                    pos[1] = best_pos[1] + mso_xorshift_double(rng_state, 0.0, 1.0) * (bounds[3] - best_pos[1]);
                } else {
                    pos[0] = best_pos[0] - mso_xorshift_double(rng_state, 0.0, 1.0) * (best_pos[0] - bounds[0]);
                    pos[1] = best_pos[1] - mso_xorshift_double(rng_state, 0.0, 1.0) * (best_pos[1] - bounds[2]);
                }
            } else { // Exploitation
                pos[0] = best_pos[0] + mso_xorshift_normal(rng_state, 0.0, 1.0) * (bounds[1] - bounds[0]) / MSO_PERTURBATION_SCALE;
                pos[1] = best_pos[1] + mso_xorshift_normal(rng_state, 0.0, 1.0) * (bounds[3] - bounds[2]) / MSO_PERTURBATION_SCALE;
            }
        } else {
            // General case for arbitrary dimensions
            for (int j = 0; j < dim; j++) {
                double rand_val = mso_xorshift_double(rng_state, 0.0, 1.0);
                if (rand_val < p_explore_threshold) { // Exploration
                    double r = mso_xorshift_double(rng_state, 0.0, 1.0);
                    if (r < 0.5) {
                        pos[j] = best_pos[j] + mso_xorshift_double(rng_state, 0.0, 1.0) * (bounds[2 * j + 1] - best_pos[j]);
                    } else {
                        pos[j] = best_pos[j] - mso_xorshift_double(rng_state, 0.0, 1.0) * (best_pos[j] - bounds[2 * j]);
                    }
                } else { // Exploitation
                    pos[j] = best_pos[j] + mso_xorshift_normal(rng_state, 0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]) / MSO_PERTURBATION_SCALE;
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void MSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    MSO_XorshiftState rng_state;
    mso_xorshift_init(&rng_state, (unsigned long)time(NULL));

    // Initialize population fitness
    double *best_pos = opt->best_solution.position;
    double best_fitness = INFINITY;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            for (int j = 0; j < dim; j++) {
                best_pos[j] = opt->population[i].position[j];
            }
        }
    }
    opt->best_solution.fitness = best_fitness;

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        mso_update_positions(opt, iter, &rng_state);

        // Evaluate fitness and update best solution
        best_fitness = opt->best_solution.fitness;
        for (int i = 0; i < pop_size; i++) {
            double *pos = opt->population[i].position;
            double fitness = objective_function(pos);
            opt->population[i].fitness = fitness;
            if (fitness < best_fitness) {
                best_fitness = fitness;
                for (int j = 0; j < dim; j++) {
                    best_pos[j] = pos[j];
                }
            }
        }
        opt->best_solution.fitness = best_fitness;

        // Optional: Print iteration progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, best_fitness);
    }
}
