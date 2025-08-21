/* DRA.c - Extreme Speed CPU-Only Implementation for Divine Religions Algorithm */
#include "DRA.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

// Fast Xorshift128+ random number generator
static uint64_t xorshift_state[2] = {1, 2};

void init_xorshift(uint64_t seed) {
    xorshift_state[0] = seed ? seed : (uint64_t)time(NULL);
    xorshift_state[1] = xorshift_state[0] ^ 0x123456789ABCDEFULL;
}

static inline uint64_t xorshift128plus() {
    uint64_t x = xorshift_state[0];
    uint64_t const y = xorshift_state[1];
    xorshift_state[0] = y;
    x ^= x << 23;
    xorshift_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return xorshift_state[1] + y;
}

static inline double rand_double(double min, double max) {
    double r = (double)xorshift128plus() / UINT64_MAX;
    return min + r * (max - min);
}

// Quicksort for sorting fitness values (inlined for small arrays)
static void quicksort_fitness(double *fitness, int *indices, int low, int high) {
    if (high - low < 10) { // Insertion sort for small arrays
        for (int i = low + 1; i <= high; i++) {
            int key = indices[i];
            double key_val = fitness[key];
            int j = i - 1;
            while (j >= low && fitness[indices[j]] > key_val) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = key;
        }
        return;
    }
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

// Centralized bound enforcement
static inline void enforce_bounds(double *position, const double *bounds, int dim) {
    for (int j = 0; j < dim; j++) {
        position[j] = fmax(bounds[2 * j], fmin(bounds[2 * j + 1], position[j]));
    }
}

// Initialize Belief Profiles
void initialize_belief_profiles(Optimizer *opt, ObjectiveFunction objective_function) {
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sort_order = (int *)malloc(opt->population_size * sizeof(int));

    // Generate initial population and compute fitness
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        temp_fitness[i] = objective_function(pos);
        temp_fitness[i] = isnan(temp_fitness[i]) ? INFINITY : temp_fitness[i];
        sort_order[i] = i;
    }

    // Sort using quicksort
    quicksort_fitness(temp_fitness, sort_order, 0, opt->population_size - 1);

    // Reorder population in-place
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = opt->population[sort_order[i]].position;
        temp_pop[i].fitness = temp_fitness[sort_order[i]];
    }
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].position = temp_pop[i].position;
        opt->population[i].fitness = temp_pop[i].fitness;
    }

    // Set best solution
    opt->best_solution.fitness = opt->population[0].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[0].position[j];
    }

    free(temp_pop);
    free(temp_fitness);
    free(sort_order);

    enforce_bound_constraints(opt);
}

// Initialize Groups (minimal implementation)
void initialize_groups(Optimizer *opt) {
    // Implicit grouping: first NUM_GROUPS are missionaries, rest are followers
}

// Miracle Operator
void miracle_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    double *new_position = (double *)malloc(opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        double rand_val = rand_double(0.0, 1.0);
        double *pos = opt->population[i].position;

        if (rand_val <= 0.5) {
            double factor = cos(PI / 2.0) * (rand_double(0.0, 1.0) - cos(rand_double(0.0, 1.0)));
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = pos[j] * factor;
            }
        } else {
            double r = rand_double(0.0, 1.0);
            double round_pow = round(pow(1.0, r));
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = pos[j] + r * (pos[j] - round_pow * pos[j]);
            }
        }

        enforce_bounds(new_position, opt->bounds, opt->dim);

        double new_fitness = objective_function(new_position);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
        if (new_fitness < opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = new_position[j];
            }
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
        double rand_val = rand_double(0.0, 1.0);
        double *pos = opt->population[i].position;

        if (rand_val > (1.0 - MIRACLE_RATE)) {
            double sum = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                sum += pos[j];
            }
            double mean_bp = sum / opt->dim;
            double sin_term = sin(sin(PI * rand_double(0.0, 1.0)));
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = pos[j] * 0.01 + mean_bp * (1.0 - MIRACLE_RATE) +
                                 (1.0 - mean_bp) - (rand_double(0.0, 1.0) - 4.0 * sin_term);
            }
        } else {
            double cos_term = cos(rand_double(0.0, 1.0));
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = leader[j] * (rand_double(0.0, 1.0) - cos_term);
            }
        }

        enforce_bounds(new_position, opt->bounds, opt->dim);

        double new_fitness = objective_function(new_position);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
        if (new_fitness < opt->population[i].fitness) {
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = new_position[j];
            }
            opt->population[i].fitness = new_fitness;
        }
    }
    free(new_position);
}

// Reward or Penalty Operator
void reward_penalty_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int index = (int)(rand_double(0, opt->population_size));
    double *pos = opt->population[index].position;
    double *new_position = (double *)malloc(opt->dim * sizeof(double));

    double rand_val = rand_double(0.0, 1.0);
    double factor = (rand_val >= REWARD_PENALTY_RATE) ? (1.0 - rand_double(-1.0, 1.0)) : (1.0 + rand_double(-1.0, 1.0));
    for (int j = 0; j < opt->dim; j++) {
        new_position[j] = pos[j] * factor;
    }

    enforce_bounds(new_position, opt->bounds, opt->dim);

    double new_fitness = objective_function(new_position);
    new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
    if (new_fitness < opt->population[index].fitness) {
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = new_position[j];
        }
        opt->population[index].fitness = new_fitness;
    }
    free(new_position);
}

// Replacement Operator
void replacement_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int num_groups = NUM_GROUPS < opt->population_size ? NUM_GROUPS : opt->population_size;
    double *temp = (double *)malloc(opt->dim * sizeof(double));

    for (int k = 0; k < num_groups; k++) {
        int follower_idx = (int)(rand_double(num_groups, opt->population_size));
        if (follower_idx < opt->population_size) {
            double *pos_k = opt->population[k].position;
            double *pos_f = opt->population[follower_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                temp[j] = pos_k[j];
                pos_k[j] = pos_f[j];
                pos_f[j] = temp[j];
            }
            opt->population[k].fitness = objective_function(pos_k);
            opt->population[follower_idx].fitness = objective_function(pos_f);
        }
    }
    free(temp);
}

// Main Optimization Function
void DRA_optimize(void *opt_ptr, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_ptr;
    init_xorshift((uint64_t)time(NULL));

    initialize_belief_profiles(opt, objective_function);
    initialize_groups(opt);

    double *new_follower = (double *)malloc(opt->dim * sizeof(double));
    double max_iter_inv = 1.0 / opt->max_iter; // Precompute for miracle_rate

    for (int iter = 0; iter < opt->max_iter; iter++) {
        double miracle_rate = rand_double(0.0, 1.0) * (1.0 - (iter * max_iter_inv * 2.0)) * rand_double(0.0, 1.0);

        // Find best and worst indices in one pass
        int min_idx = 0, worst_idx = 0;
        double min_fitness = opt->population[0].fitness;
        double max_fitness = min_fitness;
        for (int i = 1; i < opt->population_size; i++) {
            double fitness = opt->population[i].fitness;
            if (fitness < min_fitness) {
                min_fitness = fitness;
                min_idx = i;
            }
            if (fitness > max_fitness) {
                max_fitness = fitness;
                worst_idx = i;
            }
        }

        // Create new follower
        for (int j = 0; j < opt->dim; j++) {
            new_follower[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        double new_fitness = objective_function(new_follower);
        new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;

        // Belief Profile Consideration
        if (rand_double(0.0, 1.0) <= BELIEF_PROFILE_RATE) {
            int rand_idx = (int)(rand_double(0, opt->population_size));
            int rand_dim = (int)(rand_double(0, opt->dim));
            new_follower[rand_dim] = opt->population[rand_idx].position[rand_dim];
        }

        // Exploration or Exploitation
        if (rand_double(0.0, 1.0) <= miracle_rate) {
            miracle_operator(opt, objective_function);
        } else {
            double *leader = opt->population[min_idx].position;
            double sin_term = sin(rand_double(0.0, 1.0));
            for (int j = 0; j < opt->dim; j++) {
                new_follower[j] = leader[j] * (rand_double(0.0, 1.0) - sin_term);
            }
            enforce_bounds(new_follower, opt->bounds, opt->dim);
            new_fitness = objective_function(new_follower);
            new_fitness = isnan(new_fitness) ? INFINITY : new_fitness;
            proselytism_operator(opt, objective_function);
        }

        // Update worst solution
        if (new_fitness < opt->population[worst_idx].fitness) {
            double *pos = opt->population[worst_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = new_follower[j];
            }
            opt->population[worst_idx].fitness = new_fitness;
        }

        reward_penalty_operator(opt, objective_function);
        replacement_operator(opt, objective_function);

        // Update best solution
        if (opt->population[min_idx].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[min_idx].fitness;
            double *pos = opt->population[min_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = pos[j];
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
    free(new_follower);
}
