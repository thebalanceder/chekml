#include "SMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Beta distribution approximation (simplified for C)
double beta_distribution() {
    // Simplified beta(2,5) using uniform random numbers
    double u1 = rand_double(0.0, 1.0);
    double u2 = rand_double(0.0, 1.0);
    return u1 * u2; // Rough approximation for beta distribution
}

// Initialize SMO data
void initialize_smo(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    smo->num_groups = 1;
    smo->groups = (Group *)malloc(sizeof(Group));
    smo->groups[0].size = opt->population_size;
    smo->groups[0].members = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        smo->groups[0].members[i] = i;
    }
    smo->groups[0].leader_position = (double *)malloc(opt->dim * sizeof(double));
    smo->groups[0].leader_fitness = INFINITY;
    smo->groups[0].leader_count = 0;
    smo->global_leader = (double *)malloc(opt->dim * sizeof(double));
    smo->global_leader_fitness = INFINITY;
    smo->global_leader_count = 0;

    // Initialize population and find initial leaders
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = smo->objective_function(opt->population[i].position);
        if (opt->population[i].fitness < smo->groups[0].leader_fitness) {
            smo->groups[0].leader_fitness = opt->population[i].fitness;
            memcpy(smo->groups[0].leader_position, opt->population[i].position, opt->dim * sizeof(double));
        }
        if (opt->population[i].fitness < smo->global_leader_fitness) {
            smo->global_leader_fitness = opt->population[i].fitness;
            memcpy(smo->global_leader, opt->population[i].position, opt->dim * sizeof(double));
        }
    }
}

// Free SMO data
void free_smo(SMOOptimizer *smo) {
    for (int i = 0; i < smo->num_groups; i++) {
        if (smo->groups[i].members) free(smo->groups[i].members);
        if (smo->groups[i].leader_position) free(smo->groups[i].leader_position);
    }
    free(smo->groups);
    free(smo->global_leader);
}

// Beta-Hill Climbing
void beta_hill_climbing(SMOOptimizer *smo, int idx, double *new_position, double *new_fitness) {
    Optimizer *opt = smo->base_optimizer;
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    memcpy(temp_position, new_position, opt->dim * sizeof(double));

    for (int j = 0; j < opt->dim; j++) {
        if (rand_double(0.0, 1.0) < 0.5) {
            // Beta distribution for exploration
            double beta = beta_distribution();
            temp_position[j] += BHC_DELTA * beta * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        } else {
            // Hill climbing for exploitation
            temp_position[j] += BHC_DELTA * rand_double(-1.0, 1.0);
        }
        // Clip to bounds
        temp_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], temp_position[j]));
    }

    double temp_fitness = smo->objective_function(temp_position);
    if (temp_fitness < *new_fitness) {
        memcpy(new_position, temp_position, opt->dim * sizeof(double));
        *new_fitness = temp_fitness;
    }
    free(temp_position);
}

// Local Leader Phase
void local_leader_phase(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    double *new_position = (double *)malloc(opt->dim * sizeof(double));

    for (int g = 0; g < smo->num_groups; g++) {
        for (int i = 0; i < smo->groups[g].size; i++) {
            int idx = smo->groups[g].members[i];
            if (rand_double(0.0, 1.0) > PERTURBATION_RATE) {
                int rand_idx = smo->groups[g].members[rand() % smo->groups[g].size];
                for (int j = 0; j < opt->dim; j++) {
                    new_position[j] = opt->population[idx].position[j] +
                                      (smo->groups[g].leader_position[j] - opt->population[idx].position[j]) * rand_double(0.0, 1.0) +
                                      (opt->population[idx].position[j] - opt->population[rand_idx].position[j]) * rand_double(-1.0, 1.0);
                    new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
                }
                double new_fitness = opt->population[idx].fitness;
                beta_hill_climbing(smo, idx, new_position, &new_fitness);
                if (new_fitness < opt->population[idx].fitness) {
                    memcpy(opt->population[idx].position, new_position, opt->dim * sizeof(double));
                    opt->population[idx].fitness = new_fitness;
                }
            }
        }
    }
    free(new_position);
}

// Global Leader Phase
void global_leader_phase(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    double *new_position = (double *)malloc(opt->dim * sizeof(double));
    double max_fitness = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness > max_fitness) max_fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double prob = 0.9 * (1.0 - opt->population[i].fitness / max_fitness) + 0.1;
        if (rand_double(0.0, 1.0) < prob) {
            int g = 0;
            int m;
            for (; g < smo->num_groups; g++) {
                for (m = 0; m < smo->groups[g].size; m++) {
                    if (smo->groups[g].members[m] == i) break;
                }
                if (m < smo->groups[g].size) break;
            }
            int rand_idx = smo->groups[g].members[rand() % smo->groups[g].size];
            for (int j = 0; j < opt->dim; j++) {
                new_position[j] = opt->population[i].position[j] +
                                  (smo->global_leader[j] - opt->population[i].position[j]) * rand_double(0.0, 1.0) +
                                  (opt->population[rand_idx].position[j] - opt->population[i].position[j]) * rand_double(-1.0, 1.0);
                new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
            }
            double new_fitness = opt->population[i].fitness;
            beta_hill_climbing(smo, i, new_position, &new_fitness);
            if (new_fitness < opt->population[i].fitness) {
                memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
                opt->population[i].fitness = new_fitness;
            }
        }
    }
    free(new_position);
}

// Local Leader Decision
void local_leader_decision(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    for (int g = 0; g < smo->num_groups; g++) {
        int best_idx = smo->groups[g].members[0];
        double best_fitness = opt->population[best_idx].fitness;
        for (int i = 1; i < smo->groups[g].size; i++) {
            int idx = smo->groups[g].members[i];
            if (opt->population[idx].fitness < best_fitness) {
                best_idx = idx;
                best_fitness = opt->population[idx].fitness;
            }
        }
        if (best_fitness < smo->groups[g].leader_fitness) {
            smo->groups[g].leader_fitness = best_fitness;
            memcpy(smo->groups[g].leader_position, opt->population[best_idx].position, opt->dim * sizeof(double));
            smo->groups[g].leader_count = 0;
        } else {
            smo->groups[g].leader_count++;
            if (smo->groups[g].leader_count > LOCAL_LEADER_LIMIT) {
                for (int i = 0; i < smo->groups[g].size; i++) {
                    int idx = smo->groups[g].members[i];
                    double new_fitness = opt->population[idx].fitness;
                    beta_hill_climbing(smo, idx, opt->population[idx].position, &new_fitness);
                    opt->population[idx].fitness = new_fitness;
                }
                smo->groups[g].leader_count = 0;
            }
        }
    }
}

// Global Leader Decision
void global_leader_decision(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    int min_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    if (opt->population[min_idx].fitness < smo->global_leader_fitness) {
        smo->global_leader_fitness = opt->population[min_idx].fitness;
        memcpy(smo->global_leader, opt->population[min_idx].position, opt->dim * sizeof(double));
        smo->global_leader_count = 0;
    } else {
        smo->global_leader_count++;
    }

    if (smo->global_leader_count > GLOBAL_LEADER_LIMIT) {
        smo->global_leader_count = 0;
        if (smo->num_groups < MAX_GROUPS) {
            int largest_idx = 0;
            for (int g = 1; g < smo->num_groups; g++) {
                if (smo->groups[g].size > smo->groups[largest_idx].size) {
                    largest_idx = g;
                }
            }
            if (smo->groups[largest_idx].size > 1) {
                // Split largest group
                Group *new_groups = (Group *)malloc((smo->num_groups + 1) * sizeof(Group));
                memcpy(new_groups, smo->groups, smo->num_groups * sizeof(Group));
                new_groups[smo->num_groups].members = (int *)malloc(smo->groups[largest_idx].size * sizeof(int));
                new_groups[smo->num_groups].leader_position = (double *)malloc(opt->dim * sizeof(double));
                new_groups[smo->num_groups].leader_fitness = INFINITY;
                new_groups[smo->num_groups].leader_count = 0;

                int split_point = smo->groups[largest_idx].size / 2;
                new_groups[smo->num_groups].size = smo->groups[largest_idx].size - split_point;
                new_groups[largest_idx].size = split_point;
                memcpy(new_groups[smo->num_groups].members, smo->groups[largest_idx].members + split_point, new_groups[smo->num_groups].size * sizeof(int));
                new_groups[smo->num_groups].leader_fitness = opt->population[new_groups[smo->num_groups].members[0]].fitness;
                memcpy(new_groups[smo->num_groups].leader_position, opt->population[new_groups[smo->num_groups].members[0]].position, opt->dim * sizeof(double));

                free(smo->groups);
                smo->groups = new_groups;
                smo->num_groups++;
            }
        } else {
            // Merge all groups
            Group *new_groups = (Group *)malloc(sizeof(Group));
            new_groups[0].size = opt->population_size;
            new_groups[0].members = (int *)malloc(opt->population_size * sizeof(int));
            for (int i = 0; i < opt->population_size; i++) {
                new_groups[0].members[i] = i;
            }
            new_groups[0].leader_position = (double *)malloc(opt->dim * sizeof(double));
            memcpy(new_groups[0].leader_position, smo->global_leader, opt->dim * sizeof(double));
            new_groups[0].leader_fitness = smo->global_leader_fitness;
            new_groups[0].leader_count = 0;

            free_smo(smo);
            smo->groups = new_groups;
            smo->num_groups = 1;
        }
    }
}

// Main Optimization Function
void SMO_optimize(void *opt, ObjectiveFunction objective_function) {
    Optimizer *base_opt = (Optimizer *)opt;
    SMOOptimizer smo;
    smo.base_optimizer = base_opt;
    smo.objective_function = objective_function;
    initialize_smo(&smo);

    for (int iter = 0; iter < base_opt->max_iter; iter++) {
        local_leader_phase(&smo);
        global_leader_phase(&smo);
        local_leader_decision(&smo);
        global_leader_decision(&smo);

        // Update best solution
        for (int i = 0; i < base_opt->population_size; i++) {
            if (base_opt->population[i].fitness < base_opt->best_solution.fitness) {
                base_opt->best_solution.fitness = base_opt->population[i].fitness;
                memcpy(base_opt->best_solution.position, base_opt->population[i].position, base_opt->dim * sizeof(double));
            }
        }
        enforce_bound_constraints(base_opt);
        printf("Iteration %d: Best Value = %f\n", iter + 1, smo.global_leader_fitness);
    }

    free_smo(&smo);
}
