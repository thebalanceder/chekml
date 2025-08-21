#include "SMO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Beta distribution approximation (simplified for C)
inline double beta_distribution() {
    // Simplified beta(2,5) using uniform random numbers
    return rand_double(0.0, 1.0) * rand_double(0.0, 1.0); // Faster approximation
}

// Initialize SMO data
void initialize_smo(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    smo->num_groups = 1;
    smo->groups = (Group *)calloc(MAX_GROUPS, sizeof(Group)); // Pre-allocate max groups
    smo->groups[0].size = pop_size;
    smo->groups[0].members = (int *)malloc(pop_size * sizeof(int));
    smo->groups[0].leader_position = (double *)malloc(dim * sizeof(double));
    smo->groups[0].leader_fitness = INFINITY;
    smo->groups[0].leader_count = 0;
    smo->global_leader = (double *)malloc(dim * sizeof(double));
    smo->global_leader_fitness = INFINITY;
    smo->global_leader_count = 0;
    smo->temp_position = (double *)malloc(dim * sizeof(double)); // Reusable temp array

    for (int i = 0; i < pop_size; i++) {
        smo->groups[0].members[i] = i;
    }

    // Initialize population and find initial leaders
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = smo->objective_function(opt->population[i].position);
        if (opt->population[i].fitness < smo->groups[0].leader_fitness) {
            smo->groups[0].leader_fitness = opt->population[i].fitness;
            for (int j = 0; j < dim; j++) {
                smo->groups[0].leader_position[j] = opt->population[i].position[j];
            }
        }
        if (opt->population[i].fitness < smo->global_leader_fitness) {
            smo->global_leader_fitness = opt->population[i].fitness;
            for (int j = 0; j < dim; j++) {
                smo->global_leader[j] = opt->population[i].position[j];
            }
        }
    }
}

// Free SMO data
void free_smo(SMOOptimizer *smo) {
    for (int i = 0; i < smo->num_groups; i++) {
        free(smo->groups[i].members);
        free(smo->groups[i].leader_position);
    }
    free(smo->groups);
    free(smo->global_leader);
    free(smo->temp_position);
}

// Beta-Hill Climbing
void beta_hill_climbing(SMOOptimizer *smo, int idx, double *new_position, double *new_fitness) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    double *temp_position = smo->temp_position;

    for (int j = 0; j < dim; j++) {
        temp_position[j] = new_position[j];
        if (rand_double(0.0, 1.0) < 0.5) {
            // Beta distribution for exploration
            temp_position[j] += BHC_DELTA * beta_distribution() * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        } else {
            // Hill climbing for exploitation
            temp_position[j] += BHC_DELTA * rand_double(-1.0, 1.0);
        }
        temp_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], temp_position[j]));
    }

    double temp_fitness = smo->objective_function(temp_position);
    if (temp_fitness < *new_fitness) {
        for (int j = 0; j < dim; j++) {
            new_position[j] = temp_position[j];
        }
        *new_fitness = temp_fitness;
    }
}

// Local Leader Phase
void local_leader_phase(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    double *new_position = smo->temp_position;

    for (int g = 0; g < smo->num_groups; g++) {
        Group *group = &smo->groups[g];
        for (int i = 0; i < group->size; i++) {
            int idx = group->members[i];
            if (rand_double(0.0, 1.0) > PERTURBATION_RATE) {
                int rand_idx = group->members[rand() % group->size];
                double *pop_idx = opt->population[idx].position;
                double *pop_rand = opt->population[rand_idx].position;
                double *leader_pos = group->leader_position;

                for (int j = 0; j < dim; j++) {
                    new_position[j] = pop_idx[j] +
                                      (leader_pos[j] - pop_idx[j]) * rand_double(0.0, 1.0) +
                                      (pop_idx[j] - pop_rand[j]) * rand_double(-1.0, 1.0);
                    new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
                }
                double new_fitness = opt->population[idx].fitness;
                beta_hill_climbing(smo, idx, new_position, &new_fitness);
                if (new_fitness < opt->population[idx].fitness) {
                    for (int j = 0; j < dim; j++) {
                        opt->population[idx].position[j] = new_position[j];
                    }
                    opt->population[idx].fitness = new_fitness;
                }
            }
        }
    }
}

// Global Leader Phase
void global_leader_phase(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    double *new_position = smo->temp_position;

    double max_fitness = -INFINITY;
    for (int i = 0; i < pop_size; i++) {
        if (opt->population[i].fitness > max_fitness) max_fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < pop_size; i++) {
        double prob = 0.9 * (1.0 - opt->population[i].fitness / max_fitness) + 0.1;
        if (rand_double(0.0, 1.0) < prob) {
            int g = 0, m;
            for (; g < smo->num_groups; g++) {
                for (m = 0; m < smo->groups[g].size; m++) {
                    if (smo->groups[g].members[m] == i) break;
                }
                if (m < smo->groups[g].size) break;
            }
            int rand_idx = smo->groups[g].members[rand() % smo->groups[g].size];
            double *pop_i = opt->population[i].position;
            double *pop_rand = opt->population[rand_idx].position;

            for (int j = 0; j < dim; j++) {
                new_position[j] = pop_i[j] +
                                  (smo->global_leader[j] - pop_i[j]) * rand_double(0.0, 1.0) +
                                  (pop_rand[j] - pop_i[j]) * rand_double(-1.0, 1.0);
                new_position[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_position[j]));
            }
            double new_fitness = opt->population[i].fitness;
            beta_hill_climbing(smo, i, new_position, &new_fitness);
            if (new_fitness < opt->population[i].fitness) {
                for (int j = 0; j < dim; j++) {
                    opt->population[i].position[j] = new_position[j];
                }
                opt->population[i].fitness = new_fitness;
            }
        }
    }
}

// Local Leader Decision
void local_leader_decision(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;

    for (int g = 0; g < smo->num_groups; g++) {
        Group *group = &smo->groups[g];
        int best_idx = group->members[0];
        double best_fitness = opt->population[best_idx].fitness;

        for (int i = 1; i < group->size; i++) {
            int idx = group->members[i];
            if (opt->population[idx].fitness < best_fitness) {
                best_idx = idx;
                best_fitness = opt->population[idx].fitness;
            }
        }
        if (best_fitness < group->leader_fitness) {
            group->leader_fitness = best_fitness;
            for (int j = 0; j < dim; j++) {
                group->leader_position[j] = opt->population[best_idx].position[j];
            }
            group->leader_count = 0;
        } else {
            group->leader_count++;
            if (group->leader_count > LOCAL_LEADER_LIMIT) {
                for (int i = 0; i < group->size; i++) {
                    int idx = group->members[i];
                    double new_fitness = opt->population[idx].fitness;
                    beta_hill_climbing(smo, idx, opt->population[idx].position, &new_fitness);
                    opt->population[idx].fitness = new_fitness;
                }
                group->leader_count = 0;
            }
        }
    }
}

// Global Leader Decision
void global_leader_decision(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    int min_idx = 0;
    double min_fitness = opt->population[0].fitness;
    for (int i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < min_fitness) {
            min_idx = i;
            min_fitness = opt->population[i].fitness;
        }
    }
    if (min_fitness < smo->global_leader_fitness) {
        smo->global_leader_fitness = min_fitness;
        for (int j = 0; j < dim; j++) {
            smo->global_leader[j] = opt->population[min_idx].position[j];
        }
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
                Group *largest = &smo->groups[largest_idx];
                int split_point = largest->size / 2;
                smo->groups[smo->num_groups].members = (int *)malloc((largest->size - split_point) * sizeof(int));
                smo->groups[smo->num_groups].leader_position = (double *)malloc(dim * sizeof(double));
                smo->groups[smo->num_groups].size = largest->size - split_point;
                smo->groups[smo->num_groups].leader_fitness = INFINITY;
                smo->groups[smo->num_groups].leader_count = 0;

                for (int i = 0; i < smo->groups[smo->num_groups].size; i++) {
                    smo->groups[smo->num_groups].members[i] = largest->members[split_point + i];
                }
                largest->size = split_point;

                // Update new group's leader
                for (int i = 0; i < smo->groups[smo->num_groups].size; i++) {
                    int idx = smo->groups[smo->num_groups].members[i];
                    if (opt->population[idx].fitness < smo->groups[smo->num_groups].leader_fitness) {
                        smo->groups[smo->num_groups].leader_fitness = opt->population[idx].fitness;
                        for (int j = 0; j < dim; j++) {
                            smo->groups[smo->num_groups].leader_position[j] = opt->population[idx].position[j];
                        }
                    }
                }
                smo->num_groups++;
            }
        } else {
            // Merge all groups
            Group *new_groups = (Group *)calloc(1, sizeof(Group));
            new_groups->size = pop_size;
            new_groups->members = (int *)malloc(pop_size * sizeof(int));
            new_groups->leader_position = (double *)malloc(dim * sizeof(double));
            for (int i = 0; i < pop_size; i++) {
                new_groups->members[i] = i;
            }
            for (int j = 0; j < dim; j++) {
                new_groups->leader_position[j] = smo->global_leader[j];
            }
            new_groups->leader_fitness = smo->global_leader_fitness;
            new_groups->leader_count = 0;

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

    double prev_global_fitness = INFINITY;
    for (int iter = 0; iter < base_opt->max_iter; iter++) {
        local_leader_phase(&smo);
        global_leader_phase(&smo);
        local_leader_decision(&smo);
        global_leader_decision(&smo);

        // Update best solution
        for (int i = 0; i < base_opt->population_size; i++) {
            if (base_opt->population[i].fitness < base_opt->best_solution.fitness) {
                base_opt->best_solution.fitness = base_opt->population[i].fitness;
                for (int j = 0; j < base_opt->dim; j++) {
                    base_opt->best_solution.position[j] = base_opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(base_opt);

        // Early termination check
        if (fabs(smo.global_leader_fitness - prev_global_fitness) < CONVERGENCE_THRESHOLD) {
            printf("Converged at iteration %d: Best Value = %f\n", iter + 1, smo.global_leader_fitness);
            break;
        }
        prev_global_fitness = smo.global_leader_fitness;
        printf("Iteration %d: Best Value = %f\n", iter + 1, smo.global_leader_fitness);
    }

    free_smo(&smo);
}
