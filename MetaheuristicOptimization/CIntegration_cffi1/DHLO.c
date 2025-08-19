#include "DHLO.h"
#include <stdint.h>
#include <string.h>

// Fast splitmix64 RNG
static uint64_t rng_state = 0x123456789abcdef0ULL;

static void rng_seed(uint64_t seed) {
    rng_state = seed ? seed : 0x123456789abcdef0ULL;
}

static inline double rand_double(double min, double max) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t x = rng_state * 0x2545f4914f6cdd1dULL;
    return min + (max - min) * ((double)(x >> 12) / 0x1p52);
}

// Initialize Population
int dhlo_initialize_population(Optimizer *opt, int num_leaders, Solution **leaders, Solution **pbest) {
    if (!opt || !leaders || !pbest || !opt->bounds || !opt->population) return -1;

    // Initialize search agents
    for (int i = 0; i < opt->population_size; i++) {
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
    if (!*leaders) return -1;
    for (int i = 0; i < num_leaders; i++) {
        (*leaders)[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!(*leaders)[i].position) {
            for (int j = 0; j < i; j++) free((*leaders)[j].position);
            free(*leaders);
            return -1;
        }
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            (*leaders)[i].position[j] = lb + (ub - lb) * rand_double(0.0, 1.0);
        }
        (*leaders)[i].fitness = INFINITY;
    }

    // Initialize personal bests
    *pbest = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!*pbest) {
        for (int i = 0; i < num_leaders; i++) free((*leaders)[i].position);
        free(*leaders);
        return -1;
    }
    for (int i = 0; i < opt->population_size; i++) {
        (*pbest)[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!(*pbest)[i].position) {
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
    return 0;
}

// Update Leaders and Personal Bests
int dhlo_update_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int num_leaders, Solution *leaders, Solution *pbest) {
    if (!opt || !leader_fitness_history || !leaders || !pbest) return -1;

    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (isinf(fitness) || isnan(fitness)) continue;
        double *sa_pos = opt->population[i].position;

        // Update personal best
        if (fitness < pbest[i].fitness) {
            pbest[i].fitness = fitness;
            memcpy(pbest[i].position, sa_pos, opt->dim * sizeof(double));
        }

        // Update leaders
        int j = num_leaders - 1;
        for (int k = 0; k < num_leaders; k++) {
            if (fitness < leaders[k].fitness) {
                leaders[k].fitness = fitness;
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
    if (!opt || !leader_fitness_history || !num_leaders) return -1;

    int n_L = *num_leaders;
    double tol_iter = opt->max_iter * TOLERANCE_PERCENT / 100.0;

    if (DHLO_VARIANT[0] == 'V' && DHLO_VARIANT[1] == '1') {
        double AA = 1.0 - ((double)iter / (opt->max_iter / 2.0));
        if (AA > 0) {
            n_L = (int)round(3.0 + AA * (INITIAL_LEADERS - 3));
        } else {
            AA = 1.0 - ((double)iter / opt->max_iter);
            n_L = (int)round(1.0 + AA * 4.0);
        }
    } else if (DHLO_VARIANT[0] == 'V' && DHLO_VARIANT[1] == '2') {
        double AA = 1.0 - ((double)iter / opt->max_iter);
        n_L = (int)round(1.0 + AA * (INITIAL_LEADERS - 1));
    } else if (DHLO_VARIANT[0] == 'V' && DHLO_VARIANT[1] == '3') {
        double AA = -((double)iter * 10.0 / opt->max_iter);
        n_L = (int)round(INITIAL_LEADERS * exp(AA) + 1.0);
    } else {  // V4
        if (iter >= tol_iter + 1) {
            double curr_fit = leader_fitness_history[n_L - 1];
            if (curr_fit == INFINITY) {
                n_L--;
            } else {
                int prev_idx = (n_L - 1) + (iter - (int)tol_iter) * INITIAL_LEADERS;
                if (prev_idx < INITIAL_LEADERS * opt->max_iter) {
                    double diff = curr_fit - leader_fitness_history[prev_idx];
                    if ((diff < 0.0 ? diff > -1e-5 : diff < 1e-5)) {
                        n_L--;
                    }
                }
            }
        }
    }

    *num_leaders = n_L < 1 ? 1 : n_L > INITIAL_LEADERS ? INITIAL_LEADERS : n_L;
    return 0;
}

// Update Search Agents' Positions
int dhlo_update_positions(Optimizer *opt, int iter, int num_leaders, Solution *leaders, Solution *pbest, double *temp_pos, double *velocity) {
    if (!opt || !leaders || !pbest || !temp_pos || !velocity) return -1;

    if (POSITION_UPDATE_STRATEGY[0] == 'G' && POSITION_UPDATE_STRATEGY[1] == 'W') {  // GWO
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
                    double lp = leaders[k].position[j];
                    double D_alpha = fabs(C1 * lp - pos[j]);
                    XX_sum += lp - A1 * D_alpha;
                }
                temp_pos[j] = XX_sum / num_leaders;
            }
            check_bounds(temp_pos, opt->bounds, opt->dim);
            memcpy(pos, temp_pos, opt->dim * sizeof(double));
        }
    } else if (POSITION_UPDATE_STRATEGY[0] == 'P' && POSITION_UPDATE_STRATEGY[1] == 'S') {  // PSO
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
    } else if (POSITION_UPDATE_STRATEGY[0] == 'R' && POSITION_UPDATE_STRATEGY[1] == 'W') {  // RW
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
    } else {  // Fallback to GWO
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
                    double lp = leaders[k].position[j];
                    double D_alpha = fabs(C1 * lp - pos[j]);
                    XX_sum += lp - A1 * D_alpha;
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
    if (!opt || !objective_function) return -1;

    // Initialize random seed
    rng_seed((uint64_t)objective_function);

    // Initialize best solution
    if (isinf(opt->best_solution.fitness)) {
        opt->best_solution.fitness = INFINITY;
        for (int i = 0; i < opt->dim; i++) {
            opt->best_solution.position[i] = opt->bounds[2 * i];
        }
    }

    int num_leaders = INITIAL_LEADERS;
    double *leader_fitness_history = (double *)malloc(INITIAL_LEADERS * opt->max_iter * sizeof(double));
    if (!leader_fitness_history) return -1;
    double *temp_pos = (double *)malloc(opt->dim * sizeof(double));
    if (!temp_pos) {
        free(leader_fitness_history);
        return -1;
    }
    double *velocity = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    if (!velocity) {
        free(temp_pos);
        free(leader_fitness_history);
        return -1;
    }
    Solution *leaders = NULL;
    Solution *pbest = NULL;

    // Initialize history and velocity
    for (int i = 0; i < INITIAL_LEADERS * opt->max_iter; i++) {
        leader_fitness_history[i] = INFINITY;
    }
    for (int i = 0; i < opt->population_size * opt->dim; i++) {
        velocity[i] = 0.0;
    }

    if (dhlo_initialize_population(opt, num_leaders, &leaders, &pbest) != 0) {
        free(velocity);
        free(temp_pos);
        free(leader_fitness_history);
        return -1;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate population fitness
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = isinf(fitness) || isnan(fitness) ? INFINITY : fitness;
        }

        if (dhlo_update_leaders(opt, iter, leader_fitness_history + iter * INITIAL_LEADERS, num_leaders, leaders, pbest) != 0) {
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
        }
    }

    // Clean up
    for (int i = 0; i < num_leaders; i++) free(leaders[i].position);
    for (int i = 0; i < opt->population_size; i++) free(pbest[i].position);
    free(leaders);
    free(pbest);
    free(velocity);
    free(temp_pos);
    free(leader_fitness_history);

    return 0;
}
