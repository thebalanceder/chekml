#include "BHA.h"
#include <string.h>  // For memcpy
#include <time.h>    // For initial seed

// Initialize FastRNG with a seed
static inline void init_rng(FastRNG *restrict rng) {
    rng->state = (uint64_t)time(NULL) ^ 0x123456789ABCDEFULL;
}

// Initialize star positions and evaluate fitness
void initialize_stars(Optimizer *restrict opt, double (*objective_function)(double *)) {
    FastRNG rng;
    init_rng(&rng);
    double *restrict bounds = opt->bounds;
    int dim = opt->dim;
    int pop_size = opt->population_size;
    double best_fitness = INFINITY;
    int best_idx = 0;

    for (int i = 0; i < pop_size; i++) {
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = lb + fast_rand_double_bha(&rng, 0.0, 1.0) * (ub - lb);
        }
        double fitness = objective_function(pos);
        opt->population[i].fitness = fitness;
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = i;
        }
    }
    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, opt->population[best_idx].position, dim * sizeof(double));
    enforce_bound_constraints(opt);
}

// Update star positions based on black hole attraction
void update_star_positions(Optimizer *restrict opt, int black_hole_idx) {
    FastRNG rng;
    init_rng(&rng);
    double *restrict best_pos = opt->best_solution.position;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        if (i == black_hole_idx) continue;
        double *restrict pos = opt->population[i].position;
        for (int j = 0; j < dim; j += 4) {  // Loop unrolling
            if (j < dim) pos[j] += fast_rand_double_bha(&rng, 0.0, 1.0) * (best_pos[j] - pos[j]);
            if (j + 1 < dim) pos[j + 1] += fast_rand_double_bha(&rng, 0.0, 1.0) * (best_pos[j + 1] - pos[j + 1]);
            if (j + 2 < dim) pos[j + 2] += fast_rand_double_bha(&rng, 0.0, 1.0) * (best_pos[j + 2] - pos[j + 2]);
            if (j + 3 < dim) pos[j + 3] += fast_rand_double_bha(&rng, 0.0, 1.0) * (best_pos[j + 3] - pos[j + 3]);
        }
    }
    enforce_bound_constraints(opt);
}

// Replace black hole if a star has better fitness
void replace_with_better_black_hole(Optimizer *restrict opt, int *restrict black_hole_idx) {
    double best_fitness = opt->best_solution.fitness;
    int new_best_idx = *black_hole_idx;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            new_best_idx = i;
        }
    }
    if (new_best_idx != *black_hole_idx) {
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, opt->population[new_best_idx].position, dim * sizeof(double));
        *black_hole_idx = new_best_idx;
    }
}

// Replace stars that cross the event horizon
void new_star_generation(Optimizer *restrict opt, int black_hole_idx, double (*objective_function)(double *)) {
    FastRNG rng;
    init_rng(&rng);
    double total_fitness = 0.0;
    int pop_size = opt->population_size;
    int dim = opt->dim;
    double *restrict bounds = opt->bounds;
    double *restrict best_pos = opt->best_solution.position;

    // Compute total fitness
    for (int i = 0; i < pop_size; i++) {
        total_fitness += opt->population[i].fitness;
    }
    double R_squared = (opt->best_solution.fitness / total_fitness) * (opt->best_solution.fitness / total_fitness);

    for (int i = 0; i < pop_size; i++) {
        if (i == black_hole_idx) continue;
        double *restrict pos = opt->population[i].position;
        double dist_squared = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = best_pos[j] - pos[j];
            dist_squared += diff * diff;
        }
        if (dist_squared < R_squared) {
            for (int j = 0; j < dim; j++) {
                double lb = bounds[2 * j];
                double ub = bounds[2 * j + 1];
                pos[j] = lb + fast_rand_double_bha(&rng, 0.0, 1.0) * (ub - lb);
            }
            opt->population[i].fitness = objective_function(pos);
        }
    }
    enforce_bound_constraints(opt);
}

// Main optimization function
void BHA_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    int black_hole_idx = 0;
    initialize_stars(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_star_positions(opt, black_hole_idx);
        
        // Update fitness values
        for (int i = 0; i < opt->population_size; i++) {
            if (i != black_hole_idx) {  // Skip black hole to avoid redundant computation
                opt->population[i].fitness = objective_function(opt->population[i].position);
            }
        }
        replace_with_better_black_hole(opt, &black_hole_idx);
        new_star_generation(opt, black_hole_idx, objective_function);
    }
}
