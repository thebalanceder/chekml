/* DRA.c - Optimized Implementation for Divine Religions Algorithm (DRA) */
#include "DRA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

// Fast random number generator (Xorshift)
static uint32_t xorshift_state = 1;
void init_xorshift(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

uint32_t xorshift32_dra() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

double rand_double_dra(double min, double max) {
    double r = (double)xorshift32_dra() / UINT32_MAX;
    return min + r * (max - min);
}

// Quicksort for sorting fitness values
void quicksort_fitness(double *fitness, int *indices, int low, int high) {
    if (low < high) {
        double pivot = fitness[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (fitness[indices[j]] <= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        int pi = i + 1;
        quicksort_fitness(fitness, indices, low, pi - 1);
        quicksort_fitness(fitness, indices, pi + 1, high);
    }
}

// Initialize Belief Profiles
void initialize_belief_profiles(Optimizer *opt, ObjectiveFunction objective_function) {
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sort_order = (int *)malloc(opt->population_size * sizeof(int));
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));

    // Pre-allocate position arrays
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
    }

    // Generate initial population and compute fitness
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[i].position[j] = rand_double_dra(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        temp_fitness[i] = objective_function(temp_pop[i].position);
        temp_fitness[i] = isnan(temp_fitness[i]) ? INFINITY : temp_fitness[i];
        sort_order[i] = i;
    }

    // Sort using quicksort
    quicksort_fitness(temp_fitness, sort_order, 0, opt->population_size - 1);

    // Reorder population
    for (int i = 0; i < opt->population_size; i++) {
        int idx = sort_order[i];
        memcpy(opt->population[i].position, temp_pop[idx].position, opt->dim * sizeof(double));
        opt->population[i].fitness = temp_fitness[idx];
    }

    // Set best solution
    opt->best_solution.fitness = opt->population[0].fitness;
    memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));

    // Clean up
    for (int i = 0; i < opt->population_size; i++) {
        free(temp_pop[i].position);
    }
    free(temp_pop);
    free(temp_fitness);
    free(sort_order);

    enforce_bound_constraints(opt);
}

// Initialize Groups
void initialize_groups(Optimizer *opt) {
    int num_groups = NUM_GROUPS < opt->population_size ? NUM_GROUPS : opt->population_size;
    // Simplified: Mark first num_groups as missionaries, rest as followers
    // No explicit shuffling needed since grouping is implicit in replacement_operator
}

// Miracle Operator
void miracle_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    double *new_position = (double *)malloc(opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        double rand_val = rand_double_dra(0.0, 1.0);
        memcpy(new_position, opt->population[i].position, opt->dim * sizeof(double));

        if (rand_val <= 0.5) {
            double factor = cos(PI / 2.0) * (rand_double_dra(0.0, 1.0) - cos(rand_double_dra(0.0, 1.0)));
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] *= factor;
            }
        } else {
            double r = rand_double_dra(0.0, 1.0);
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] += r * (new_position[j] - round(pow(1.0, r)) * new_position[j]);
            }
        }

        // Enforce bounds
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
        }

        // Update fitness if improved
        double new_fitness = objective_function(new_position);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
        if (new_fitness < opt->population[i].fitness) {
            memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
            opt->population[i].fitness = new_fitness;
        }
    }
    free(new_position);
}

// Proselytism Operator
void proselytism_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int min_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    double *leader = opt->population[min_idx].position;
    double *new_position = (double *)malloc(opt->dim * sizeof(double));

    for (int i = 0; i < opt->population_size; i++) {
        double rand_val = rand_double_dra(0.0, 1.0);
        memcpy(new_position, opt->population[i].position, opt->dim * sizeof(double));

        if (rand_val > (1.0 - MIRACLE_RATE)) {
            double sum = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                sum += new_position[j];
            }
            double mean_bp = sum / opt->dim;
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = (new_position[j] * 0.01 +
                                  mean_bp * (1.0 - MIRACLE_RATE) +
                                  (1.0 - mean_bp) -
                                  (rand_double_dra(0.0, 1.0) - 4.0 * sin(sin(PI * rand_double_dra(0.0, 1.0)))));
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = leader[j] * (rand_double_dra(0.0, 1.0) - cos(rand_double_dra(0.0, 1.0)));
            }
        }

        // Enforce bounds
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
        }

        // Update fitness if improved
        double new_fitness = objective_function(new_position);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
        if (new_fitness < opt->population[i].fitness) {
            memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
            opt->population[i].fitness = new_fitness;
        }
    }
    free(new_position);
}

// Reward or Penalty Operator
void reward_penalty_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int index = (int)(rand_double_dra(0, opt->population_size));
    double *new_position = (double *)malloc(opt->dim * sizeof(double));
    memcpy(new_position, opt->population[index].position, opt->dim * sizeof(double));

    double rand_val = rand_double_dra(0.0, 1.0);
    double factor = (rand_val >= REWARD_PENALTY_RATE) ? (1.0 - rand_double_dra(-1.0, 1.0)) : (1.0 + rand_double_dra(-1.0, 1.0));
    for (int j = 0; j < opt->dim; j++) {
        new_position[j] *= factor;
    }

    // Enforce bounds
    for (int j = 0; j < opt->dim; j++) {
        new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
    }

    // Update fitness if improved
    double new_fitness = objective_function(new_position);
    new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
    if (new_fitness < opt->population[index].fitness) {
        memcpy(opt->population[index].position, new_position, opt->dim * sizeof(double));
        opt->population[index].fitness = new_fitness;
    }
    free(new_position);
}

// Replacement Operator
void replacement_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int num_groups = NUM_GROUPS < opt->population_size ? NUM_GROUPS : opt->population_size;
    double *temp = (double *)malloc(opt->dim * sizeof(double));

    for (int k = 0; k < num_groups; k++) {
        int follower_idx = (int)(rand_double_dra(num_groups, opt->population_size));
        if (follower_idx < opt->population_size) {
            memcpy(temp, opt->population[k].position, opt->dim * sizeof(double));
            memcpy(opt->population[k].position, opt->population[follower_idx].position, opt->dim * sizeof(double));
            memcpy(opt->population[follower_idx].position, temp, opt->dim * sizeof(double));

            opt->population[k].fitness = objective_function(opt->population[k].position);
            opt->population[follower_idx].fitness = objective_function(opt->population[follower_idx].position);
        }
    }
    free(temp);
}

// Main Optimization Function
void DRA_optimize(void *opt_ptr, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_ptr;
    init_xorshift((uint32_t)time(NULL));

    initialize_belief_profiles(opt, objective_function);
    initialize_groups(opt);

    double *new_follower = (double *)malloc(opt->dim * sizeof(double));
    for (int iter = 0; iter < opt->max_iter; iter++) {
        double miracle_rate = rand_double_dra(0.0, 1.0) * (1.0 - ((double)iter / opt->max_iter * 2.0)) * rand_double_dra(0.0, 1.0);

        // Find best and worst indices
        int min_idx = 0, worst_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
            if (opt->population[i].fitness > opt->population[worst_idx].fitness) {
                worst_idx = i;
            }
        }

        // Create new follower
        for (int j = 0; j < opt->dim; j++) {
            new_follower[j] = rand_double_dra(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        double new_fitness = objective_function(new_follower);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;

        // Belief Profile Consideration
        if (rand_double_dra(0.0, 1.0) <= BELIEF_PROFILE_RATE) {
            int rand_idx = (int)(rand_double_dra(0, opt->population_size));
            int rand_dim = (int)(rand_double_dra(0, opt->dim));
            new_follower[rand_dim] = opt->population[rand_idx].position[rand_dim];
        }

        // Exploration or Exploitation
        if (rand_double_dra(0.0, 1.0) <= miracle_rate) {
            miracle_operator(opt, objective_function);
        } else {
            for (int j = 0; j < opt->dim; j++) {
                new_follower[j] = opt->population[min_idx].position[j] * (rand_double_dra(0.0, 1.0) - sin(rand_double_dra(0.0, 1.0)));
            }
            for (int j = 0; j < opt->dim; j++) {
                new_follower[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_follower[j]));
            }
            new_fitness = objective_function(new_follower);
            new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
            proselytism_operator(opt, objective_function);
        }

        // Update worst solution
        if (new_fitness < opt->population[worst_idx].fitness) {
            memcpy(opt->population[worst_idx].position, new_follower, opt->dim * sizeof(double));
            opt->population[worst_idx].fitness = new_fitness;
        }

        reward_penalty_operator(opt, objective_function);
        replacement_operator(opt, objective_function);

        // Update best solution
        if (opt->population[min_idx].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[min_idx].fitness;
            memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
    free(new_follower);
}
