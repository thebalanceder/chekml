#include "PuO.h"
#include <string.h>
#include <time.h>

// Fast Xorshift RNG
static unsigned int xorshift_state = 1;
static void xorshift_seed(unsigned int seed) {
    xorshift_state = seed ? seed : 1;
}

static unsigned int xorshift32() {
    unsigned int x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

static double rand_double(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / 0xFFFFFFFFu);
}

// Quicksort for sorting indices by fitness
static void quicksort_indices(double *fitness, int *indices, int low, int high) {
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
        quicksort_indices(fitness, indices, low, pi - 1);
        quicksort_indices(fitness, indices, pi + 1, high);
    }
}

void initialize_pumas(Optimizer *opt, PumaOptimizerState *state) {
    if (opt->dim <= 0 || opt->population_size <= 0) {
        fprintf(stderr, "Error: Invalid dimensions or population size\n");
        return;
    }
    xorshift_seed((unsigned int)time(NULL));
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_puma_bounds(opt);
}

void evaluate_pumas(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

void enforce_puma_bounds(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double pos = opt->population[i].position[j];
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = pos > ub ? ub : (pos < lb ? lb : pos);
        }
    }
}

void puma_exploration_phase(Optimizer *opt, PumaOptimizerState *state, double (*objective_function)(double *)) {
    double pCR = PCR_INITIAL;
    double p = (1.0 - pCR) / opt->population_size;
    double *temp_position = state->temp_position;
    double *y = state->y;
    double *z = state->z;
    double *fitness = state->fitness;
    int *indices = state->indices;
    int *A = state->perm;

    // Compute fitness and sort indices
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }
    quicksort_indices(fitness, indices, 0, opt->population_size - 1);

    for (int i = 0; i < opt->population_size; i++) {
        int idx = indices[i];
        memcpy(temp_position, opt->population[idx].position, opt->dim * sizeof(double));

        // Generate permutation excluding current index
        for (int k = 0; k < opt->population_size; k++) A[k] = k;
        for (int k = opt->population_size - 1; k > 0; k--) {
            int swap_idx = xorshift32() % (k + 1);
            int temp = A[k];
            A[k] = A[swap_idx];
            A[swap_idx] = temp;
        }
        int a_idx = 0;
        while (A[a_idx] == idx && a_idx < opt->population_size) a_idx++;
        int a = A[a_idx];
        int b = A[(a_idx + 1) % opt->population_size];
        int c = A[(a_idx + 2) % opt->population_size];
        int d = A[(a_idx + 3) % opt->population_size];
        int e = A[(a_idx + 4) % opt->population_size];
        int f = A[(a_idx + 5) % opt->population_size];

        double G = 2.0 * rand_double(0.0, 1.0) - 1.0;
        if (rand_double(0.0, 1.0) < 0.5) {
            for (int j = 0; j < opt->dim; j++) {
                y[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                y[j] = opt->population[a].position[j] + G * (opt->population[a].position[j] - opt->population[b].position[j]) +
                       G * ((opt->population[a].position[j] - opt->population[b].position[j]) -
                            (opt->population[c].position[j] - opt->population[d].position[j]) +
                            (opt->population[c].position[j] - opt->population[d].position[j]) -
                            (opt->population[e].position[j] - opt->population[f].position[j]));
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            double yj = y[j];
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            y[j] = yj > ub ? ub : (yj < lb ? lb : yj);
        }

        memcpy(z, temp_position, opt->dim * sizeof(double));
        int j0 = xorshift32() % opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            if (j == j0 || rand_double(0.0, 1.0) <= pCR) {
                z[j] = y[j];
            }
        }

        double new_fitness = objective_function(z);
        if (new_fitness < opt->population[idx].fitness) {
            memcpy(opt->population[idx].position, z, opt->dim * sizeof(double));
            opt->population[idx].fitness = new_fitness;
        } else {
            pCR += p;
        }
    }
}

void puma_exploitation_phase(Optimizer *opt, PumaOptimizerState *state, int iter) {
    double T = (double)opt->max_iter;
    double *beta2 = state->beta2;
    double *w = state->w;
    double *v = state->v;
    double *F1 = state->F1;
    double *F2 = state->F2;
    double *S1 = state->S1;
    double *S2 = state->S2;
    double *VEC = state->VEC;
    double *Xatack = state->Xatack;
    double *mbest = state->mbest;

    // Compute mean position
    memset(mbest, 0, opt->dim * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            mbest[j] += opt->population[i].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        mbest[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double beta1 = 2.0 * rand_double(0.0, 1.0);
        double rand_val = rand_double(0.0, 1.0);
        for (int j = 0; j < opt->dim; j++) {
            beta2[j] = rand_double(-1.0, 1.0);
            w[j] = rand_double(-1.0, 1.0);
            v[j] = rand_double(-1.0, 1.0);
            F1[j] = rand_double(-1.0, 1.0) * exp(2.0 - iter * (2.0 / T));
            F2[j] = w[j] * v[j] * v[j] * cos(2.0 * rand_val * w[j]);
            S1[j] = 2.0 * rand_double(0.0, 1.0) - 1.0 + rand_double(-1.0, 1.0);
            S2[j] = F1[j] * (2.0 * rand_double(0.0, 1.0) - 1.0) * opt->population[i].position[j] +
                    F2[j] * (1.0 - (2.0 * rand_double(0.0, 1.0) - 1.0)) * opt->best_solution.position[j];
            VEC[j] = S2[j] / S1[j];
        }

        if (rand_val <= 0.5) {
            memcpy(Xatack, VEC, opt->dim * sizeof(double));
            if (rand_double(0.0, 1.0) > state->q_probability) {
                int r1 = xorshift32() % opt->population_size;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = opt->best_solution.position[j] +
                                                    beta1 * exp(beta2[j]) * (opt->population[r1].position[j] - opt->population[i].position[j]);
                }
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = beta1 * Xatack[j] - opt->best_solution.position[j];
                }
            }
        } else {
            int r1 = 1 + (xorshift32() % (opt->population_size - 1));
            double sign = (xorshift32() % 2) ? 1.0 : -1.0;
            double denom = 1.0 + (state->beta * rand_double(0.0, 1.0));
            for (int j = 0; j < opt->dim; j++) { // Fixed typo: j% to j++
                opt->population[i].position[j] = (mbest[j] * opt->population[r1].position[j] - sign * opt->population[i].position[j]) / denom;
            }
        }
    }
    enforce_puma_bounds(opt);
}

void PuO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (opt->dim <= 0 || opt->population_size <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid optimizer parameters\n");
        return;
    }

    PumaOptimizerState state = {0};
    state.q_probability = Q_PROBABILITY;
    state.beta = BETA_FACTOR;
    state.PF[0] = PF1;
    state.PF[1] = PF2;
    state.PF[2] = PF3;
    state.mega_explore = MEGA_EXPLORE_INIT;
    state.mega_exploit = MEGA_EXPLOIT_INIT;
    state.unselected[0] = 1.0;
    state.unselected[1] = 1.0;
    for (int i = 0; i < 3; i++) {
        state.seq_time_explore[i] = 1.0;
        state.seq_time_exploit[i] = 1.0;
        state.seq_cost_explore[i] = 1.0;
        state.seq_cost_exploit[i] = 1.0;
    }
    state.flag_change = 1;

    // Preallocate buffers
    state.temp_position = (double *)malloc(opt->dim * sizeof(double));
    state.y = (double *)malloc(opt->dim * sizeof(double));
    state.z = (double *)malloc(opt->dim * sizeof(double));
    state.fitness = (double *)malloc(opt->population_size * sizeof(double));
    state.indices = (int *)malloc(opt->population_size * sizeof(int));
    state.perm = (int *)malloc(opt->population_size * sizeof(int));
    state.beta2 = (double *)malloc(opt->dim * sizeof(double));
    state.w = (double *)malloc(opt->dim * sizeof(double));
    state.v = (double *)malloc(opt->dim * sizeof(double));
    state.F1 = (double *)malloc(opt->dim * sizeof(double));
    state.F2 = (double *)malloc(opt->dim * sizeof(double));
    state.S1 = (double *)malloc(opt->dim * sizeof(double));
    state.S2 = (double *)malloc(opt->dim * sizeof(double));
    state.VEC = (double *)malloc(opt->dim * sizeof(double));
    state.Xatack = (double *)malloc(opt->dim * sizeof(double));
    state.mbest = (double *)malloc(opt->dim * sizeof(double));

    initialize_pumas(opt, &state);
    evaluate_pumas(opt, objective_function);
    int min_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[min_idx].fitness;
    memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
    double initial_best_cost = opt->best_solution.fitness;

    double costs_explore[3];
    double costs_exploit[3];

    // Unexperienced Phase
    for (int iter = 0; iter < 3; iter++) {
        puma_exploration_phase(opt, &state, objective_function);
        evaluate_pumas(opt, objective_function);
        double min_fitness = opt->population[0].fitness;
        min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }
        costs_explore[iter] = min_fitness;

        puma_exploitation_phase(opt, &state, iter + 1);
        evaluate_pumas(opt, objective_function);
        min_fitness = opt->population[0].fitness;
        min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }
        costs_exploit[iter] = min_fitness;

        opt->best_solution.fitness = opt->population[min_idx].fitness;
        memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
    }

    // Hyper Initialization
    state.seq_cost_explore[0] = fabs(initial_best_cost - costs_explore[0]);
    state.seq_cost_exploit[0] = fabs(initial_best_cost - costs_exploit[0]);
    state.seq_cost_explore[1] = fabs(costs_explore[1] - costs_explore[0]);
    state.seq_cost_exploit[1] = fabs(costs_exploit[1] - costs_exploit[0]);
    state.seq_cost_explore[2] = fabs(costs_explore[2] - costs_explore[1]);
    state.seq_cost_exploit[2] = fabs(costs_exploit[2] - costs_exploit[1]);

    for (int i = 0; i < 3; i++) {
        if (state.seq_cost_explore[i] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.seq_cost_explore[i];
        }
        if (state.seq_cost_exploit[i] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.seq_cost_exploit[i];
        }
    }
    if (state.pf_f3_size == 0) {
        state.pf_f3[state.pf_f3_size++] = 1e-10;
    }

    double f1_explore = state.PF[0] * (state.seq_cost_explore[0] / state.seq_time_explore[0]);
    double f1_exploit = state.PF[0] * (state.seq_cost_exploit[0] / state.seq_time_exploit[0]);
    double sum_cost_explore = state.seq_cost_explore[0] + state.seq_cost_explore[1] + state.seq_cost_explore[2];
    double sum_cost_exploit = state.seq_cost_exploit[0] + state.seq_cost_exploit[1] + state.seq_cost_exploit[2];
    double sum_time_explore = state.seq_time_explore[0] + state.seq_time_explore[1] + state.seq_time_explore[2];
    double sum_time_exploit = state.seq_time_exploit[0] + state.seq_time_exploit[1] + state.seq_time_exploit[2];
    double f2_explore = state.PF[1] * (sum_cost_explore / sum_time_explore);
    double f2_exploit = state.PF[1] * (sum_cost_exploit / sum_time_exploit);
    state.score_explore = (state.PF[0] * f1_explore) + (state.PF[1] * f2_explore);
    state.score_exploit = (state.PF[0] * f1_exploit) + (state.PF[1] * f2_exploit);

    // Experienced Phase
    double *t_best = (double *)malloc(opt->dim * sizeof(double));
    for (int iter = 3; iter < opt->max_iter; iter++) {
        int select_flag;
        double t_best_cost;

        if (state.score_explore > state.score_exploit) {
            select_flag = 1;
            puma_exploration_phase(opt, &state, objective_function);
            state.unselected[1] += 1.0;
            state.unselected[0] = 1.0;
            state.f3_explore = state.PF[2];
            state.f3_exploit += state.PF[2];
            evaluate_pumas(opt, objective_function);
            min_idx = 0;
            for (int i = 1; i < opt->population_size; i++) {
                if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                    min_idx = i;
                }
            }
            t_best_cost = opt->population[min_idx].fitness;
            memcpy(t_best, opt->population[min_idx].position, opt->dim * sizeof(double));
            state.seq_cost_explore[2] = state.seq_cost_explore[1];
            state.seq_cost_explore[1] = state.seq_cost_explore[0];
            state.seq_cost_explore[0] = fabs(opt->best_solution.fitness - t_best_cost);
            if (state.seq_cost_explore[0] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
                state.pf_f3[state.pf_f3_size++] = state.seq_cost_explore[0];
            }
            if (t_best_cost < opt->best_solution.fitness) {
                opt->best_solution.fitness = t_best_cost;
                memcpy(opt->best_solution.position, t_best, opt->dim * sizeof(double));
            }
        } else {
            select_flag = 2;
            puma_exploitation_phase(opt, &state, iter + 1);
            state.unselected[0] += 1.0;
            state.unselected[1] = 1.0;
            state.f3_explore += state.PF[2];
            state.f3_exploit = state.PF[2];
            evaluate_pumas(opt, objective_function);
            min_idx = 0;
            for (int i = 1; i < opt->population_size; i++) {
                if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                    min_idx = i;
                }
            }
            t_best_cost = opt->population[min_idx].fitness;
            memcpy(t_best, opt->population[min_idx].position, opt->dim * sizeof(double));
            state.seq_cost_exploit[2] = state.seq_cost_exploit[1];
            state.seq_cost_exploit[1] = state.seq_cost_exploit[0];
            state.seq_cost_exploit[0] = fabs(opt->best_solution.fitness - t_best_cost);
            if (state.seq_cost_exploit[0] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
                state.pf_f3[state.pf_f3_size++] = state.seq_cost_exploit[0];
            }
            if (t_best_cost < opt->best_solution.fitness) {
                opt->best_solution.fitness = t_best_cost;
                memcpy(opt->best_solution.position, t_best, opt->dim * sizeof(double));
            }
        }

        if (state.flag_change != select_flag) {
            state.flag_change = select_flag;
            state.seq_time_explore[2] = state.seq_time_explore[1];
            state.seq_time_explore[1] = state.seq_time_explore[0];
            state.seq_time_explore[0] = state.unselected[0];
            state.seq_time_exploit[2] = state.seq_time_exploit[1];
            state.seq_time_exploit[1] = state.seq_time_exploit[0];
            state.seq_time_exploit[0] = state.unselected[1];
        }

        double f1_explore = state.PF[0] * (state.seq_cost_explore[0] / state.seq_time_explore[0]);
        double f1_exploit = state.PF[0] * (state.seq_cost_exploit[0] / state.seq_time_exploit[0]);
        double f2_explore = state.PF[1] * (sum_cost_explore / sum_time_explore);
        double f2_exploit = state.PF[1] * (sum_cost_exploit / sum_time_exploit);

        if (state.score_explore < state.score_exploit) {
            state.mega_explore = state.mega_explore > 0.01 ? state.mega_explore - 0.01 : 0.01;
            state.mega_exploit = MEGA_EXPLOIT_INIT;
        } else {
            state.mega_explore = MEGA_EXPLORE_INIT;
            state.mega_exploit = state.mega_exploit > 0.01 ? state.mega_exploit - 0.01 : 0.01;
        }

        double lmn_explore = 1.0 - state.mega_explore;
        double lmn_exploit = 1.0 - state.mega_exploit;
        double min_pf_f3 = state.pf_f3[0];
        for (int i = 1; i < state.pf_f3_size; i++) {
            if (state.pf_f3[i] < min_pf_f3) min_pf_f3 = state.pf_f3[i];
        }
        state.score_explore = state.mega_explore * f1_explore + state.mega_explore * f2_explore + lmn_explore * (min_pf_f3 * state.f3_explore);
        state.score_exploit = state.mega_exploit * f1_exploit + state.mega_exploit * f2_exploit + lmn_exploit * (min_pf_f3 * state.f3_exploit);
    }

    free(state.temp_position);
    free(state.y);
    free(state.z);
    free(state.fitness);
    free(state.indices);
    free(state.perm);
    free(state.beta2);
    free(state.w);
    free(state.v);
    free(state.F1);
    free(state.F2);
    free(state.S1);
    free(state.S2);
    free(state.VEC);
    free(state.Xatack);
    free(state.mbest);
    free(t_best);
}
