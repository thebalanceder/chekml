#include "DHLO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

// Xorshift random number generator for better performance
static uint32_t xorshift_state = 1;

static void xorshift_seed(uint32_t seed) {
    xorshift_state = seed ? seed : (uint32_t)time(NULL);
}

static uint32_t xorshift32() {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static double rand_double(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / (double)0xFFFFFFFF);
}

// Initialize Population (Search Agents, Leaders, and Personal Bests)
int dhlo_initialize_population(Optimizer *opt, int num_leaders, Solution **leaders, Solution **pbest) {
    if (!opt || !leaders || !pbest || !opt->bounds || !opt->population) {
        fprintf(stderr, "Error: Null pointer in dhlo_initialize_population\n");
        return -1;
    }

    fprintf(stderr, "Initializing population: pop_size=%d, dim=%d, num_leaders=%d\n", opt->population_size, opt->dim, num_leaders);

    // Initialize search agents
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: Null population position at index %d\n", i);
            return -1;
        }
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }

    // Initialize leaders
    *leaders = (Solution *)malloc(num_leaders * sizeof(Solution));
    if (!*leaders) {
        fprintf(stderr, "Error: Failed to allocate leaders\n");
        return -1;
    }
    for (int i = 0; i < num_leaders; i++) {
        (*leaders)[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!(*leaders)[i].position) {
            fprintf(stderr, "Error: Failed to allocate leader position %d\n", i);
            for (int j = 0; j < i; j++) free((*leaders)[j].position);
            free(*leaders);
            return -1;
        }
        double *pos = (*leaders)[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        (*leaders)[i].fitness = INFINITY;
    }

    // Initialize personal bests for PSO
    *pbest = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!*pbest) {
        fprintf(stderr, "Error: Failed to allocate pbest\n");
        for (int i = 0; i < num_leaders; i++) free((*leaders)[i].position);
        free(*leaders);
        return -1;
    }
    for (int i = 0; i < opt->population_size; i++) {
        (*pbest)[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!(*pbest)[i].position) {
            fprintf(stderr, "Error: Failed to allocate pbest position %d\n", i);
            for (int j = 0; j < i; j++) free((*pbest)[j].position);
            free(*pbest);
            for (int j = 0; j < num_leaders; j++) free((*leaders)[j].position);
            free(*leaders);
            return -1;
        }
        memcpy((*pbest)[i].position, opt->population[i].position, opt->dim * sizeof(double));
        (*pbest)[i].fitness = INFINITY;
    }

    enforce_bound_constraints(opt);
    fprintf(stderr, "Population initialized successfully\n");
    return 0;
}

// Update Leaders and Personal Bests
int dhlo_update_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int num_leaders, Solution *leaders, Solution *pbest) {
    if (!opt || !leader_fitness_history || !leaders || !pbest) {
        fprintf(stderr, "Error: Null pointer in dhlo_update_leaders\n");
        return -1;
    }

    fprintf(stderr, "Updating leaders: iter=%d, num_leaders=%d\n", iter, num_leaders);

    for (int i = 0; i < opt->population_size; i++) {
        double sa_fitness = opt->population[i].fitness;
        if (isinf(sa_fitness) || isnan(sa_fitness)) {
            fprintf(stderr, "Warning: Invalid fitness %f for agent %d at iter %d\n", sa_fitness, i, iter);
            continue;
        }
        double *sa_pos = opt->population[i].position;

        // Update personal best
        if (sa_fitness < pbest[i].fitness) {
            pbest[i].fitness = sa_fitness;
            memcpy(pbest[i].position, sa_pos, opt->dim * sizeof(double));
        }

        // Update leaders
        int j = num_leaders - 1;
        for (int k = 0; k < num_leaders; k++) {
            if (sa_fitness < leaders[k].fitness) {
                leaders[k].fitness = sa_fitness;
                memcpy(leaders[k].position, sa_pos, opt->dim * sizeof(double));
                j = k;
                break;
            }
        }
        if (j < num_leaders) {
            leader_fitness_history[j] = leaders[j].fitness;
        }
    }
    enforce_bound_constraints(opt);
    return 0;
}

// Adjust Number of Leaders
int dhlo_adjust_num_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int *num_leaders) {
    if (!opt || !leader_fitness_history || !num_leaders) {
        fprintf(stderr, "Error: Null pointer in dhlo_adjust_num_leaders\n");
        return -1;
    }

    fprintf(stderr, "Adjusting num_leaders: iter=%d, variant=%s\n", iter, DHLO_VARIANT);

    int n_L = *num_leaders;
    double tol_iter = opt->max_iter * TOLERANCE_PERCENT / 100.0;

    if (strcmp(DHLO_VARIANT, "V1") == 0) {
        double AA = 1.0 - ((double)iter / (opt->max_iter / 2.0));
        if (AA > 0) {
            n_L = (int)round(3.0 + AA * (INITIAL_LEADERS - 3));
        } else {
            AA = 1.0 - ((double)iter / opt->max_iter);
            n_L = (int)round(1.0 + AA * 4.0);
        }
    } else if (strcmp(DHLO_VARIANT, "V2") == 0) {
        double AA = 1.0 - ((double)iter / opt->max_iter);
        n_L = (int)round(1.0 + AA * (INITIAL_LEADERS - 1));
    } else if (strcmp(DHLO_VARIANT, "V3") == 0) {
        double AA = -((double)iter * 10.0 / opt->max_iter);
        n_L = (int)round(INITIAL_LEADERS * exp(AA) + 1.0);
    } else if (strcmp(DHLO_VARIANT, "V4") == 0) {
        if (iter >= tol_iter + 1) {
            double curr_fit = leader_fitness_history[n_L - 1];
            if (curr_fit == INFINITY) {
                n_L--;
            } else {
                int prev_idx = n_L - 1 + (iter - (int)tol_iter) * INITIAL_LEADERS;
                if (prev_idx < INITIAL_LEADERS * opt->max_iter) {
                    double diff = curr_fit - leader_fitness_history[prev_idx];
                    if (diff < 0.0 ? diff > -1e-5 : diff < 1e-5) {
                        n_L--;
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "Invalid DHLO variant: %s. Using V4.\n", DHLO_VARIANT);
        if (iter >= tol_iter + 1) {
            double curr_fit = leader_fitness_history[n_L - 1];
            if (curr_fit == INFINITY) {
                n_L--;
            } else {
                int prev_idx = n_L - 1 + (iter - (int)tol_iter) * INITIAL_LEADERS;
                if (prev_idx < INITIAL_LEADERS * opt->max_iter) {
                    double diff = curr_fit - leader_fitness_history[prev_idx];
                    if (diff < 0.0 ? diff > -1e-5 : diff < 1e-5) {
                        n_L--;
                    }
                }
            }
        }
    }

    *num_leaders = (n_L < 1) ? 1 : (n_L > INITIAL_LEADERS) ? INITIAL_LEADERS : n_L;
    return 0;
}

// Update Search Agents' Positions
int dhlo_update_positions(Optimizer *opt, int iter, int num_leaders, Solution *leaders, Solution *pbest, double *temp_pos, double *velocity) {
    if (!opt || !leaders || !pbest || !temp_pos || !velocity) {
        fprintf(stderr, "Error: Null pointer in dhlo_update_positions\n");
        return -1;
    }

    fprintf(stderr, "Updating positions: iter=%d, strategy=%s\n", iter, POSITION_UPDATE_STRATEGY);

    if (strcmp(POSITION_UPDATE_STRATEGY, "GWO") == 0) {
        double a = A_MAX - iter * ((A_MAX - A_MIN) / opt->max_iter);
        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                double XX_sum = 0.0;
                for (int k = 0; k < num_leaders; k++) {
                    double r1 = rand_double(0.0, 1.0);
                    double r2 = rand_double(0.0, 1.0);
                    double A1 = 2.0 * a * r1 - a;
                    double C1 = 2.0 * r2;
                    double leader_pos = leaders[k].position[j];
                    double D_alpha = C1 * leader_pos - pos[j];
                    D_alpha = D_alpha < 0.0 ? -D_alpha : D_alpha;
                    XX_sum += leader_pos - A1 * D_alpha;
                }
                temp_pos[j] = XX_sum / num_leaders;
            }
            check_bounds(temp_pos, opt->bounds, opt->dim);
            memcpy(pos, temp_pos, opt->dim * sizeof(double));
        }
    } else if (strcmp(POSITION_UPDATE_STRATEGY, "PSO") == 0) {
        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            double *vel = velocity + i * opt->dim;
            int leader_idx = (int)(rand_double(0.0, num_leaders));
            for (int j = 0; j < opt->dim; j++) {
                double r1 = rand_double(0.0, 1.0);
                double r2 = rand_double(0.0, 1.0);
                vel[j] = PSO_INERTIA * vel[j] +
                         PSO_COGNITIVE * r1 * (pbest[i].position[j] - pos[j]) +
                         PSO_SOCIAL * r2 * (leaders[leader_idx].position[j] - pos[j]);
                temp_pos[j] = pos[j] + vel[j];
            }
            check_bounds(temp_pos, opt->bounds, opt->dim);
            memcpy(pos, temp_pos, opt->dim * sizeof(double));
        }
    } else if (strcmp(POSITION_UPDATE_STRATEGY, "RW") == 0) {
        double scale = RW_STEP_SCALE * (1.0 - (double)iter / opt->max_iter);
        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            int leader_idx = (int)(rand_double(0.0, num_leaders));
            for (int j = 0; j < opt->dim; j++) {
                double step = scale * rand_double(-1.0, 1.0);
                temp_pos[j] = pos[j] + step * (leaders[leader_idx].position[j] - pos[j]);
            }
            check_bounds(temp_pos, opt->bounds, opt->dim);
            memcpy(pos, temp_pos, opt->dim * sizeof(double));
        }
    } else {
        fprintf(stderr, "Invalid position update strategy: %s. Using GWO.\n", POSITION_UPDATE_STRATEGY);
        double a = A_MAX - iter * ((A_MAX - A_MIN) / opt->max_iter);
        for (int i = 0; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                double XX_sum = 0.0;
                for (int k = 0; k < num_leaders; k++) {
                    double r1 = rand_double(0.0, 1.0);
                    double r2 = rand_double(0.0, 1.0);
                    double A1 = 2.0 * a * r1 - a;
                    double C1 = 2.0 * r2;
                    double leader_pos = leaders[k].position[j];
                    double D_alpha = C1 * leader_pos - pos[j];
                    D_alpha = D_alpha < 0.0 ? -D_alpha : D_alpha;
                    XX_sum += leader_pos - A1 * D_alpha;
                }
                temp_pos[j] = XX_sum / num_leaders;
            }
            check_bounds(temp_pos, opt->bounds, opt->dim);
            memcpy(pos, temp_pos, opt->dim * sizeof(double));
        }
    }
    enforce_bound_constraints(opt);
    return 0;
}

// Main Optimization Function
int DHLO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt) {
        fprintf(stderr, "Error: Null optimizer pointer\n");
        return -1;
    }
    if (!objective_function) {
        fprintf(stderr, "Error: Null objective function pointer\n");
        return -1;
    }

    fprintf(stderr, "Starting DHLO_optimize: variant=%s, strategy=%s, dim=%d, pop_size=%d, max_iter=%d\n",
            DHLO_VARIANT, POSITION_UPDATE_STRATEGY, opt->dim, opt->population_size, opt->max_iter);

    // Initialize best solution if not set
    if (isinf(opt->best_solution.fitness)) {
        opt->best_solution.fitness = INFINITY;
        for (int i = 0; i < opt->dim; i++) {
            opt->best_solution.position[i] = opt->bounds[2 * i];  // Initialize to lower bound
        }
    }

    int num_leaders = INITIAL_LEADERS;
    double *leader_fitness_history = (double *)malloc(INITIAL_LEADERS * opt->max_iter * sizeof(double));
    if (!leader_fitness_history) {
        fprintf(stderr, "Error: Failed to allocate leader_fitness_history\n");
        return -1;
    }
    double *temp_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_pos) {
        fprintf(stderr, "Error: Failed to allocate temp_pos\n");
        free(leader_fitness_history);
        return -1;
    }
    double *velocity = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    if (!velocity) {
        fprintf(stderr, "Error: Failed to allocate velocity\n");
        free(temp_pos);
        free(leader_fitness_history);
        return -1;
    }
    Solution *leaders = NULL;
    Solution *pbest = NULL;

    // Initialize random seed for each run
    xorshift_seed((uint32_t)time(NULL));

    // Initialize history and velocity
    for (int i = 0; i < INITIAL_LEADERS * opt->max_iter; i++) {
        leader_fitness_history[i] = INFINITY;
    }
    for (int i = 0; i < opt->population_size * opt->dim; i++) {
        velocity[i] = 0.0;
    }

    if (dhlo_initialize_population(opt, num_leaders, &leaders, &pbest) != 0) {
        fprintf(stderr, "Error: Failed to initialize population\n");
        free(velocity);
        free(temp_pos);
        free(leader_fitness_history);
        return -1;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate population fitness
        for (int i = 0; i < opt->population_size; i++) {
            if (!opt->population[i].position) {
                fprintf(stderr, "Error: Null population position at iter %d, index %d\n", iter, i);
                for (int j = 0; j < num_leaders; j++) free(leaders[j].position);
                for (int j = 0; j < opt->population_size; j++) free(pbest[j].position);
                free(leaders);
                free(pbest);
                free(velocity);
                free(temp_pos);
                free(leader_fitness_history);
                return -1;
            }
            double fitness = objective_function(opt->population[i].position);
            if (isinf(fitness) || isnan(fitness)) {
                fprintf(stderr, "Warning: Invalid fitness %f for agent %d at iter %d\n", fitness, i, iter);
                opt->population[i].fitness = INFINITY;
            } else {
                opt->population[i].fitness = fitness;
            }
        }

        if (dhlo_update_leaders(opt, iter, leader_fitness_history + iter * INITIAL_LEADERS, num_leaders, leaders, pbest) != 0) {
            fprintf(stderr, "Error: Failed to update leaders at iter %d\n", iter);
            for (int j = 0; j < num_leaders; j++) free(leaders[j].position);
            for (int j = 0; j < opt->population_size; j++) free(pbest[j].position);
            free(leaders);
            free(pbest);
            free(velocity);
            free(temp_pos);
            free(leader_fitness_history);
            return -1;
        }

        if (dhlo_adjust_num_leaders(opt, iter, leader_fitness_history, &num_leaders) != 0) {
            fprintf(stderr, "Error: Failed to adjust num_leaders at iter %d\n", iter);
            for (int j = 0; j < num_leaders; j++) free(leaders[j].position);
            for (int j = 0; j < opt->population_size; j++) free(pbest[j].position);
            free(leaders);
            free(pbest);
            free(velocity);
            free(temp_pos);
            free(leader_fitness_history);
            return -1;
        }

        if (dhlo_update_positions(opt, iter, num_leaders, leaders, pbest, temp_pos, velocity) != 0) {
            fprintf(stderr, "Error: Failed to update positions at iter %d\n", iter);
            for (int j = 0; j < num_leaders; j++) free(leaders[j].position);
            for (int j = 0; j < opt->population_size; j++) free(pbest[j].position);
            free(leaders);
            free(pbest);
            free(velocity);
            free(temp_pos);
            free(leader_fitness_history);
            return -1;
        }

        // Update best solution
        if (!isinf(leaders[0].fitness) && leaders[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = leaders[0].fitness;
            memcpy(opt->best_solution.position, leaders[0].position, opt->dim * sizeof(double));
            fprintf(stderr, "Updated best_solution at iter %d: fitness=%f, position=[%f, %f]\n",
                    iter, opt->best_solution.fitness, opt->best_solution.position[0], opt->best_solution.position[1]);
        }

        fprintf(stderr, "Iteration %d: Best Value = %f, Num Leaders = %d\n", iter + 1, opt->best_solution.fitness, num_leaders);
    }

    // Clean up
    for (int i = 0; i < num_leaders; i++) {
        free(leaders[i].position);
    }
    for (int i = 0; i < opt->population_size; i++) {
        free(pbest[i].position);
    }
    free(leaders);
    free(pbest);
    free(velocity);
    free(temp_pos);
    free(leader_fitness_history);

    fprintf(stderr, "DHLO_optimize completed successfully\n");
    return 0;
}
