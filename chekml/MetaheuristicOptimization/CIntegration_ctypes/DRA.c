/* DRA.c - Implementation file for Divine Religions Algorithm (DRA) */
#include "DRA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand()
#include <time.h>    // For time() if seeding random generator
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Belief Profiles (Population)
void initialize_belief_profiles(Optimizer *opt, ObjectiveFunction objective_function) {
    double *temp_fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *sort_order = (int *)malloc(opt->population_size * sizeof(int));

    // Generate random initial positions and compute fitness
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        temp_fitness[i] = objective_function(opt->population[i].position);
        if (isnan(temp_fitness[i])) {
            temp_fitness[i] = INFINITY;
        }
        sort_order[i] = i;
    }

    // Sort by fitness (simple bubble sort for clarity)
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (temp_fitness[sort_order[j]] > temp_fitness[sort_order[j + 1]]) {
                int temp = sort_order[j];
                sort_order[j] = sort_order[j + 1];
                sort_order[j + 1] = temp;
            }
        }
    }

    // Reorder population based on sorted fitness
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(temp_pop[i].position, opt->population[sort_order[i]].position, opt->dim * sizeof(double));
        temp_pop[i].fitness = temp_fitness[sort_order[i]];
    }

    // Copy back to population
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(opt->population[i].position, temp_pop[i].position, opt->dim * sizeof(double));
        opt->population[i].fitness = temp_pop[i].fitness;
        free(temp_pop[i].position);
    }
    free(temp_pop);
    free(temp_fitness);
    free(sort_order);

    // Set best solution
    opt->best_solution.fitness = opt->population[0].fitness;
    memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));

    enforce_bound_constraints(opt);
}

// Initialize Groups (Missionaries and Followers)
void initialize_groups(Optimizer *opt) {
    int num_groups = NUM_GROUPS < opt->population_size ? NUM_GROUPS : opt->population_size;
    int num_followers = opt->population_size - num_groups;

    // Shuffle follower indices
    int *follower_indices = (int *)malloc(num_followers * sizeof(int));
    for (int i = 0; i < num_followers; i++) {
        follower_indices[i] = i + num_groups;
    }
    for (int i = num_followers - 1; i > 0; i--) {
        int j = (int)(rand_double(0, i + 1));
        int temp = follower_indices[i];
        follower_indices[i] = follower_indices[j];
        follower_indices[j] = temp;
    }

    // Assign followers to groups (simulated via random grouping)
    for (int i = 0; i < num_followers; i++) {
        int group_idx = (int)(rand_double(0, num_groups));
        // In C, we don't maintain explicit group lists; assume grouping is implicit
    }

    free(follower_indices);
}

// Miracle Operator (Exploration)
void miracle_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    for (int i = 0; i < opt->population_size; i++) {
        double rand_val = rand_double(0.0, 1.0);
        if (rand_val <= 0.5) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] *= cos(PI / 2.0) * (rand_double(0.0, 1.0) - cos(rand_double(0.0, 1.0)));
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double r = rand_double(0.0, 1.0);
                opt->population[i].position[j] += r * (opt->population[i].position[j] - round(pow(1.0, r)) * opt->population[i].position[j]);
            }
        }

        // Ensure bounds
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }

        // Update fitness
        double new_fitness = objective_function(opt->population[i].position);
        if (isnan(new_fitness)) {
            new_fitness = INFINITY;
        }
        if (new_fitness < opt->population[i].fitness) {
            opt->population[i].fitness = new_fitness;
        }
    }
}

// Proselytism Operator (Exploitation)
void proselytism_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int min_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    double *leader = opt->population[min_idx].position;

    for (int i = 0; i < opt->population_size; i++) {
        double rand_val = rand_double(0.0, 1.0);
        if (rand_val > (1.0 - MIRACLE_RATE)) {
            double sum = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                sum += opt->population[i].position[j];
            }
            double mean_bp = sum / opt->dim;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = (opt->population[i].position[j] * 0.01 +
                                                 mean_bp * (1.0 - MIRACLE_RATE) +
                                                 (1.0 - mean_bp) -
                                                 (rand_double(0.0, 1.0) - 4.0 * sin(sin(PI * rand_double(0.0, 1.0)))));
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = leader[j] * (rand_double(0.0, 1.0) - cos(rand_double(0.0, 1.0)));
            }
        }

        // Ensure bounds
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }

        // Update fitness
        double new_fitness = objective_function(opt->population[i].position);
        if (isnan(new_fitness)) {
            new_fitness = INFINITY;
        }
        if (new_fitness < opt->population[i].fitness) {
            opt->population[i].fitness = new_fitness;
        }
    }
}

// Reward or Penalty Operator
void reward_penalty_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int index = (int)(rand_double(0, opt->population_size));
    double rand_val = rand_double(0.0, 1.0);

    if (rand_val >= REWARD_PENALTY_RATE) {
        // Reward
        for (int j = 0; j < opt->dim; j++) {
            opt->population[index].position[j] *= (1.0 - rand_double(-1.0, 1.0));
        }
    } else {
        // Penalty
        for (int j = 0; j < opt->dim; j++) {
            opt->population[index].position[j] *= (1.0 + rand_double(-1.0, 1.0));
        }
    }

    // Ensure bounds
    for (int j = 0; j < opt->dim; j++) {
        if (opt->population[index].position[j] < opt->bounds[2 * j]) {
            opt->population[index].position[j] = opt->bounds[2 * j];
        } else if (opt->population[index].position[j] > opt->bounds[2 * j + 1]) {
            opt->population[index].position[j] = opt->bounds[2 * j + 1];
        }
    }

    // Update fitness
    double new_fitness = objective_function(opt->population[index].position);
    if (isnan(new_fitness)) {
        new_fitness = INFINITY;
    }
    if (new_fitness < opt->population[index].fitness) {
        opt->population[index].fitness = new_fitness;
    }
}

// Replacement Operator (Missionaries and Followers)
void replacement_operator(Optimizer *opt, ObjectiveFunction objective_function) {
    int num_groups = NUM_GROUPS < opt->population_size ? NUM_GROUPS : opt->population_size;
    for (int k = 0; k < num_groups; k++) {
        // Simulate group by picking a random follower (approximation)
        int follower_idx = (int)(rand_double(num_groups, opt->population_size));
        if (follower_idx < opt->population_size) {
            double *temp = (double *)malloc(opt->dim * sizeof(double));
            memcpy(temp, opt->population[k].position, opt->dim * sizeof(double));
            memcpy(opt->population[k].position, opt->population[follower_idx].position, opt->dim * sizeof(double));
            memcpy(opt->population[follower_idx].position, temp, opt->dim * sizeof(double));
            free(temp);

            // Update fitness for swapped positions
            opt->population[k].fitness = objective_function(opt->population[k].position);
            opt->population[follower_idx].fitness = objective_function(opt->population[follower_idx].position);
        }
    }
}

// Main Optimization Function
void DRA_optimize(void *opt_ptr, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_ptr;

    initialize_belief_profiles(opt, objective_function);
    initialize_groups(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update miracle rate
        double miracle_rate = rand_double(0.0, 1.0) * (1.0 - ((double)iter / opt->max_iter * 2.0)) * rand_double(0.0, 1.0);

        // Select leader (best belief profile)
        int min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
        }

        // Create new follower
        double *new_follower = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            new_follower[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        double new_fitness = objective_function(new_follower);
        if (isnan(new_fitness)) {
            new_fitness = INFINITY;
        }

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
            for (int j = 0; j < opt->dim; j++) {
                new_follower[j] = opt->population[min_idx].position[j] * (rand_double(0.0, 1.0) - sin(rand_double(0.0, 1.0)));
            }
            for (int j = 0; j < opt->dim; j++) {
                if (new_follower[j] < opt->bounds[2 * j]) {
                    new_follower[j] = opt->bounds[2 * j];
                } else if (new_follower[j] > opt->bounds[2 * j + 1]) {
                    new_follower[j] = opt->bounds[2 * j + 1];
                }
            }
            new_fitness = objective_function(new_follower);
            if (isnan(new_fitness)) {
                new_fitness = INFINITY;
            }
            proselytism_operator(opt, objective_function);
        }

        // Update worst solution with new follower
        int worst_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness > opt->population[worst_idx].fitness) {
                worst_idx = i;
            }
        }
        if (new_fitness < opt->population[worst_idx].fitness) {
            memcpy(opt->population[worst_idx].position, new_follower, opt->dim * sizeof(double));
            opt->population[worst_idx].fitness = new_fitness;
        }
        free(new_follower);

        // Reward or Penalty
        reward_penalty_operator(opt, objective_function);

        // Replacement
        replacement_operator(opt, objective_function);

        // Update best solution
        min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
        }
        if (opt->population[min_idx].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[min_idx].fitness;
            memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
