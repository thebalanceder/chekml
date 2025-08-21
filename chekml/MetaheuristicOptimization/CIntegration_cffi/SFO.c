#include "SFO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Fast Xorshift random number generator
static void xorshift_init(SFOXorshiftState *state, uint32_t seed) {
    state->x = seed | 1;
    state->y = 1812433253U * state->x;
    state->z = 1812433253U * state->y;
    state->w = 1812433253U * state->z;
}

static double xorshift_double(SFOXorshiftState *state, double min, double max) {
    uint32_t t = state->x ^ (state->x << 11);
    state->x = state->y; state->y = state->z; state->z = state->w;
    state->w = state->w ^ (state->w >> 19) ^ (t ^ (t >> 8));
    return min + (max - min) * (state->w / 4294967296.0);
}

// Precompute trigonometric values for efficiency
static void precompute_trig(int max_iter, double *cos_table, double *atan_table) {
    for (int t = 0; t < max_iter; t++) {
        cos_table[t] = cos(M_PI * t / (2.0 * max_iter));
        atan_table[t] = atan(pow((double)t / max_iter, 2.0 * M_PI));
    }
}

// Calculate Instruction(t) as per Eq. (1)
static double calculate_instruction(int t, int max_iter, SFOXorshiftState *state) {
    return (1.0 - 0.15 * xorshift_double(state, 0.0, 1.0)) * (1.0 - (double)t / max_iter);
}

// Calculate probability of losing contact p(t) as per Eq. (3)
static double calculate_loss_probability(int t, double *cos_table) {
    return P0 * cos_table[t];
}

// Calculate raid coefficient w(t) as per Eq. (7)
static double calculate_raid_coefficient(int t, double *atan_table){
    return 0.75 - 0.55 * atan_table[t];
}

// Exploration Phase (combines large-scale search and raid)
void sfo_exploration_phase(Optimizer *opt, int idx, double r1, double w, double *temp_pos, SFOXorshiftState *state) {
    double *pos = opt->population[idx].position;
    if (r1 >= 0.5) { // Large-scale search, Eq. (4)
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            double term1 = r1 * (opt->best_solution.position[j] - pos[j]);
            double term2 = (1.0 - r1) * (ub - lb) * (xorshift_double(state, 0.0, 1.0) < 0.5 ? -1.0 : 1.0);
            temp_pos[j] = pos[j] + term1 + term2;
        }
    } else { // Raid, Eq. (5, 6)
        int aim_idx = (idx + 1 + (int)(xorshift_double(state, 0.0, opt->population_size - 1))) % opt->population_size;
        double f_i = opt->population[idx].fitness;
        double f_aim = opt->population[aim_idx].fitness;
        double scale = (f_i + f_aim != 0.0) ? f_i / (f_i + f_aim) : 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double A_i = scale * (opt->population[aim_idx].position[j] - pos[j]);
            temp_pos[j] = pos[j] + w * A_i;
        }
    }
    memcpy(pos, temp_pos, opt->dim * sizeof(double));
}

// Transition Phase as per Eq. (8)
void transition_phase(Optimizer *opt, int idx, double r2, double w, double instruction, double *temp_pos, SFOXorshiftState *state) {
    double *pos = opt->population[idx].position;
    if (r2 >= 0.5) {
        sfo_exploration_phase(opt, idx, 0.4, w, temp_pos, state); // Reuse raid with r1 < 0.5
    } else {
        for (int j = 0; j < opt->dim; j++) {
            temp_pos[j] = instruction * (opt->best_solution.position[j] - pos[j]) + 0.1 * pos[j];
        }
        memcpy(pos, temp_pos, opt->dim * sizeof(double));
    }
}

// Arrest-Rescue Strategy as per Eq. (9, 10)
void arrest_rescue(Optimizer *opt, double *temp_pos, SFOXorshiftState *state) {
    double *X_ave = temp_pos; // Reuse temp_pos to avoid allocation
    for (int j = 0; j < opt->dim; j++) {
        X_ave[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            X_ave[j] += opt->population[i].position[j];
        }
        X_ave[j] /= opt->population_size;
    }
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        SFOXorshiftState local_state = *state;
        #ifdef _OPENMP
        xorshift_init(&local_state, state->w ^ (uint32_t)(omp_get_thread_num() + i)); // Unique seed per thread
        #endif
        for (int j = 0; j < opt->dim; j++) {
            double r = xorshift_double(&local_state, -1.0, 1.0);
            pos[j] = opt->best_solution.position[j] + r * fabs(opt->best_solution.position[j] - X_ave[j]);
        }
    }
}

// Unmanned Search as per Eq. (11, 12)
void unmanned_search(Optimizer *opt, int t, double *temp_pos, double *temp_vec, SFOXorshiftState *state) {
    double *v = temp_vec;
    double sum_squares = 0.0;
    for (int j = 0; j < opt->dim; j++) {
        double lb = opt->bounds[2 * j];
        double ub = opt->bounds[2 * j + 1];
        double c = K * (lb + (1.0 - (double)t / opt->max_iter) * (ub - lb));
        v[j] = xorshift_double(state, -1.0, 1.0);
        sum_squares += v[j] * v[j]; // ✅ Fixed syntax error
        temp_pos[j] = c; // Store c for normalization
    }
    double norm = sqrt(sum_squares);
    if (norm != 0.0) {
        for (int j = 0; j < opt->dim; j++) {
            v[j] = v[j] / norm * temp_pos[j];
        }
    }
    int base_idx = (int)xorshift_double(state, 0.0, opt->population_size);
    double *pos = opt->population[base_idx].position;
    for (int j = 0; j < opt->dim; j++) {
        pos[j] += v[j];
    }
}

// Main Optimization Function
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Preallocate buffers
    double *temp_pos = (double *)malloc(opt->dim * sizeof(double));
    double *temp_vec = (double *)malloc(opt->dim * sizeof(double));
    double *cos_table = (double *)malloc(opt->max_iter * sizeof(double));
    double *atan_table = (double *)malloc(opt->max_iter * sizeof(double));

    // Initialize random number generator
    SFOXorshiftState state;
    xorshift_init(&state, (uint32_t)time(NULL));
    precompute_trig(opt->max_iter, cos_table, atan_table);

    // Evaluate initial population
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        #pragma omp critical
        {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }
    }

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Unmanned search
        unmanned_search(opt, t, temp_pos, temp_vec, &state);
        opt->population[0].fitness = objective_function(opt->population[0].position);
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
        }

        // Calculate iteration parameters
        double instruction = calculate_instruction(t, opt->max_iter, &state);
        double p = calculate_loss_probability(t, cos_table);
        double w = calculate_raid_coefficient(t, atan_table);

        // Update population
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            SFOXorshiftState local_state = state;
            #ifdef _OPENMP
            xorshift_init(&local_state, state.w ^ (uint32_t)(omp_get_thread_num() + i + t)); // Unique seed per thread/iteration
            #endif
            if (xorshift_double(&local_state, 0.0, 1.0) < p) {
                continue; // Simulate loss of contact
            }

            double r1 = xorshift_double(&local_state, 0.0, 1.0);
            double r2 = xorshift_double(&local_state, 0.0, 1.0);

            if (instruction >= TV1) { // Exploration phase
                sfo_exploration_phase(opt, i, r1, w, temp_pos, &local_state);
            } else if (instruction > TV2) { // Transition phase
                transition_phase(opt, i, r2, w, instruction, temp_pos, &local_state);
            } else { // Exploitation phase
                arrest_rescue(opt, temp_pos, &local_state);
            }

            opt->population[i].fitness = objective_function(opt->population[i].position);
            #pragma omp critical
            {
                if (opt->population[i].fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = opt->population[i].fitness;
                    memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                }
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    // Cleanup
    free(temp_pos);
    free(temp_vec);
    free(cos_table);
    free(atan_table);
}
