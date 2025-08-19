#include "SMO.h"
#include "generaloptimizer.h"
#include <time.h>

// Fast linear congruential generator (LCG) for random numbers
#define LCG_A 1664525
#define LCG_C 1013904223
#define LCG_M 4294967296ULL // 2^32

inline double fast_rand_double(SMOOptimizer *smo) {
    smo->rng_state = (LCG_A * smo->rng_state + LCG_C) % LCG_M;
    return (double)smo->rng_state / LCG_M;
}

inline double rand_double_range(SMOOptimizer *smo, double min, double max) {
    return min + (max - min) * fast_rand_double(smo);
}

inline double beta_distribution(SMOOptimizer *smo) {
    return fast_rand_double(smo) * fast_rand_double(smo); // Simplified beta approximation
}

// Initialize SMO data
void initialize_smo(SMOOptimizer *smo) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    smo->num_groups = 1;
    smo->groups[0].size = pop_size;
    smo->groups[0].leader_fitness = INFINITY;
    smo->groups[0].leader_count = 0;
    smo->global_leader_fitness = INFINITY;
    smo->global_leader_count = 0;
    smo->rng_state = (unsigned int)time(NULL); // Seed RNG

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

// Beta-Hill Climbing (inlined for speed)
inline void beta_hill_climbing(SMOOptimizer *smo, int idx, double *new_position, double *new_fitness) {
    Optimizer *opt = smo->base_optimizer;
    const int dim = opt->dim;
    double *temp_position = smo->temp_position;

    for (int j = 0; j < dim; j++) {
        temp_position[j] = new_position[j];
        if (fast_rand_double(smo) < 0.5) {
            temp_position[j] += BHC_DELTA * beta_distribution(smo) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        } else {
            temp_position[j] += BHC_DELTA * rand_double_range(smo, -1.0, 1.0);
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

// Main Optimization Function (merged phases for minimal overhead)
void SMO_optimize(void *opt, ObjectiveFunction objective_function) {
    Optimizer *base_opt = (Optimizer *)opt;
    SMOOptimizer smo = {0};
    smo.base_optimizer = base_opt;
    smo.objective_function = objective_function;
    initialize_smo(&smo);

    const int dim = base_opt->dim;
    const int pop_size = base_opt->population_size;
    double prev_global_fitness = INFINITY;

    for (int iter = 0; iter < base_opt->max_iter; iter++) {
        // Local Leader Phase
        for (int g = 0; g < smo.num_groups; g++) {
            Group *group = &smo.groups[g];
            for (int i = 0; i < group->size; i++) {
                int idx = group->members[i];
                if (fast_rand_double(&smo) > PERTURBATION_RATE) {
                    int rand_idx = group->members[(int)(fast_rand_double(&smo) * group->size)];
                    double *pop_idx = base_opt->population[idx].position;
                    double *pop_rand = base_opt->population[rand_idx].position;
                    double *leader_pos = group->leader_position;
                    double *new_pos = smo.temp_position;

                    for (int j = 0; j < dim; j++) {
                        new_pos[j] = pop_idx[j] +
                                     (leader_pos[j] - pop_idx[j]) * fast_rand_double(&smo) +
                                     (pop_idx[j] - pop_rand[j]) * rand_double_range(&smo, -1.0, 1.0);
                        new_pos[j] = fmax(base_opt->bounds[2 * j], fmin(base_opt->bounds[2 * j + 1], new_pos[j]));
                    }
                    double new_fitness = base_opt->population[idx].fitness;
                    beta_hill_climbing(&smo, idx, new_pos, &new_fitness);
                    if (new_fitness < base_opt->population[idx].fitness) {
                        for (int j = 0; j < dim; j++) {
                            base_opt->population[idx].position[j] = new_pos[j];
                        }
                        base_opt->population[idx].fitness = new_fitness;
                    }
                }
            }
        }

        // Global Leader Phase
        double max_fitness = -INFINITY;
        for (int i = 0; i < pop_size; i++) {
            if (base_opt->population[i].fitness > max_fitness) max_fitness = base_opt->population[i].fitness;
        }

        for (int i = 0; i < pop_size; i++) {
            double prob = 0.9 * (1.0 - base_opt->population[i].fitness / max_fitness) + 0.1;
            if (fast_rand_double(&smo) < prob) {
                int g = 0, m;
                for (; g < smo.num_groups; g++) {
                    for (m = 0; m < smo.groups[g].size; m++) {
                        if (smo.groups[g].members[m] == i) break;
                    }
                    if (m < smo.groups[g].size) break;
                }
                int rand_idx = smo.groups[g].members[(int)(fast_rand_double(&smo) * smo.groups[g].size)];
                double *pop_i = base_opt->population[i].position;
                double *pop_rand = base_opt->population[rand_idx].position;
                double *new_pos = smo.temp_position;

                for (int j = 0; j < dim; j++) {
                    new_pos[j] = pop_i[j] +
                                 (smo.global_leader[j] - pop_i[j]) * fast_rand_double(&smo) +
                                 (pop_rand[j] - pop_i[j]) * rand_double_range(&smo, -1.0, 1.0);
                    new_pos[j] = fmax(base_opt->bounds[2 * j], fmin(base_opt->bounds[2 * j + 1], new_pos[j]));
                }
                double new_fitness = base_opt->population[i].fitness;
                beta_hill_climbing(&smo, i, new_pos, &new_fitness);
                if (new_fitness < base_opt->population[i].fitness) {
                    for (int j = 0; j < dim; j++) {
                        base_opt->population[i].position[j] = new_pos[j];
                    }
                    base_opt->population[i].fitness = new_fitness;
                }
            }
        }

        // Local Leader Decision
        for (int g = 0; g < smo.num_groups; g++) {
            Group *group = &smo.groups[g];
            int best_idx = group->members[0];
            double best_fitness = base_opt->population[best_idx].fitness;

            for (int i = 1; i < group->size; i++) {
                int idx = group->members[i];
                if (base_opt->population[idx].fitness < best_fitness) {
                    best_idx = idx;
                    best_fitness = base_opt->population[idx].fitness;
                }
            }
            if (best_fitness < group->leader_fitness) {
                group->leader_fitness = best_fitness;
                for (int j = 0; j < dim; j++) {
                    group->leader_position[j] = base_opt->population[best_idx].position[j];
                }
                group->leader_count = 0;
            } else {
                group->leader_count++;
                if (group->leader_count > LOCAL_LEADER_LIMIT) {
                    for (int i = 0; i < group->size; i++) {
                        int idx = group->members[i];
                        double new_fitness = base_opt->population[idx].fitness;
                        beta_hill_climbing(&smo, idx, base_opt->population[idx].position, &new_fitness);
                        base_opt->population[idx].fitness = new_fitness;
                    }
                    group->leader_count = 0;
                }
            }
        }

        // Global Leader Decision
        int min_idx = 0;
        double min_fitness = base_opt->population[0].fitness;
        for (int i = 1; i < pop_size; i++) {
            if (base_opt->population[i].fitness < min_fitness) {
                min_idx = i;
                min_fitness = base_opt->population[i].fitness;
            }
        }
        if (min_fitness < smo.global_leader_fitness) {
            smo.global_leader_fitness = min_fitness;
            for (int j = 0; j < dim; j++) {
                smo.global_leader[j] = base_opt->population[min_idx].position[j];
            }
            smo.global_leader_count = 0;
        } else {
            smo.global_leader_count++;
        }

        if (smo.global_leader_count > GLOBAL_LEADER_LIMIT) {
            smo.global_leader_count = 0;
            if (smo.num_groups < MAX_GROUPS) {
                int largest_idx = 0;
                for (int g = 1; g < smo.num_groups; g++) {
                    if (smo.groups[g].size > smo.groups[largest_idx].size) {
                        largest_idx = g;
                    }
                }
                if (smo.groups[largest_idx].size > 1) {
                    Group *largest = &smo.groups[largest_idx];
                    int split_point = largest->size / 2;
                    Group *new_group = &smo.groups[smo.num_groups];
                    new_group->size = largest->size - split_point;
                    new_group->leader_fitness = INFINITY;
                    new_group->leader_count = 0;

                    for (int i = 0; i < new_group->size; i++) {
                        new_group->members[i] = largest->members[split_point + i];
                    }
                    largest->size = split_point;

                    for (int i = 0; i < new_group->size; i++) {
                        int idx = new_group->members[i];
                        if (base_opt->population[idx].fitness < new_group->leader_fitness) {
                            new_group->leader_fitness = base_opt->population[idx].fitness;
                            for (int j = 0; j < dim; j++) {
                                new_group->leader_position[j] = base_opt->population[idx].position[j];
                            }
                        }
                    }
                    smo.num_groups++;
                }
            } else {
                Group *new_group = &smo.groups[0];
                new_group->size = pop_size;
                for (int i = 0; i < pop_size; i++) {
                    new_group->members[i] = i;
                }
                for (int j = 0; j < dim; j++) {
                    new_group->leader_position[j] = smo.global_leader[j];
                }
                new_group->leader_fitness = smo.global_leader_fitness;
                new_group->leader_count = 0;
                smo.num_groups = 1;
            }
        }

        // Update best solution
        for (int i = 0; i < pop_size; i++) {
            if (base_opt->population[i].fitness < base_opt->best_solution.fitness) {
                base_opt->best_solution.fitness = base_opt->population[i].fitness;
                for (int j = 0; j < dim; j++) {
                    base_opt->best_solution.position[j] = base_opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(base_opt);

        // Early termination
        if (fabs(smo.global_leader_fitness - prev_global_fitness) < SMO_CONVERGENCE_THRESHOLD) {
            printf("Converged at iteration %d: Best Value = %f\n", iter + 1, smo.global_leader_fitness);
            break;
        }
        prev_global_fitness = smo.global_leader_fitness;
        printf("Iteration %d: Best Value = %f\n", iter + 1, smo.global_leader_fitness);
    }
}
