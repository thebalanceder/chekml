#include "PuO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Maximum size for pf_f3 array
#define PF_F3_MAX_SIZE 6

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Puma Population
void initialize_pumas(Optimizer *opt) {
    if (opt->dim <= 0 || opt->population_size <= 0) {
        fprintf(stderr, "Error: Invalid dimensions or population size\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY; // Initialize fitness
    }
    enforce_puma_bounds(opt);
}

// Evaluate Fitness for All Pumas
void evaluate_pumas(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Enforce Boundary Constraints
void enforce_puma_bounds(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            } else if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            }
        }
    }
}

// Exploration Phase (Global Search)
void puma_exploration_phase(Optimizer *opt, PumaOptimizerState *state, double (*objective_function)(double *)) {
    double pCR = PCR_INITIAL;
    double p = (1.0 - pCR) / opt->population_size;
    double *temp_position = (double *)malloc(opt->dim * sizeof(double));
    double *y = (double *)malloc(opt->dim * sizeof(double));
    double *z = (double *)malloc(opt->dim * sizeof(double));

    // Compute fitness and sort indices
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }
    // Simple bubble sort for indices based on fitness
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (fitness[indices[j]] > fitness[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // Create sorted population
    Solution *sorted_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        sorted_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(sorted_pop[i].position, opt->population[indices[i]].position, opt->dim * sizeof(double));
        sorted_pop[i].fitness = fitness[indices[i]];
    }

    for (int i = 0; i < opt->population_size; i++) {
        memcpy(temp_position, sorted_pop[i].position, opt->dim * sizeof(double));

        // Generate permutation excluding current index
        int *A = (int *)malloc(opt->population_size * sizeof(int));
        for (int k = 0; k < opt->population_size; k++) A[k] = k;
        for (int k = opt->population_size - 1; k > 0; k--) {
            int swap_idx = rand() % (k + 1);
            int temp = A[k];
            A[k] = A[swap_idx];
            A[swap_idx] = temp;
        }
        int a_idx = 0, a, b, c, d, e, f;
        while (A[a_idx] == i && a_idx < opt->population_size) a_idx++;
        a = A[a_idx];
        b = A[(a_idx + 1) % opt->population_size];
        c = A[(a_idx + 2) % opt->population_size];
        d = A[(a_idx + 3) % opt->population_size];
        e = A[(a_idx + 4) % opt->population_size];
        f = A[(a_idx + 5) % opt->population_size];

        double G = 2.0 * rand_double(0.0, 1.0) - 1.0; // Eq 26
        if (rand_double(0.0, 1.0) < 0.5) {
            for (int j = 0; j < opt->dim; j++) {
                y[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]); // Eq 25
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                y[j] = sorted_pop[a].position[j] + G * (sorted_pop[a].position[j] - sorted_pop[b].position[j]) +
                       G * ((sorted_pop[a].position[j] - sorted_pop[b].position[j]) -
                            (sorted_pop[c].position[j] - sorted_pop[d].position[j]) +
                            (sorted_pop[c].position[j] - sorted_pop[d].position[j]) -
                            (sorted_pop[e].position[j] - sorted_pop[f].position[j])); // Eq 25
            }
        }

        // Boundary check for y
        for (int j = 0; j < opt->dim; j++) {
            if (y[j] > opt->bounds[2 * j + 1]) y[j] = opt->bounds[2 * j + 1];
            if (y[j] < opt->bounds[2 * j]) y[j] = opt->bounds[2 * j];
        }

        memcpy(z, temp_position, opt->dim * sizeof(double));
        int j0 = rand() % opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            if (j == j0 || rand_double(0.0, 1.0) <= pCR) {
                z[j] = y[j];
            }
        }

        memcpy(opt->population[i].position, z, opt->dim * sizeof(double));
        if (objective_function(opt->population[i].position) < sorted_pop[i].fitness) {
            memcpy(sorted_pop[i].position, z, opt->dim * sizeof(double));
            sorted_pop[i].fitness = objective_function(opt->population[i].position);
        } else {
            pCR += p; // Eq 30
        }

        free(A);
    }

    // Copy sorted population back
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(opt->population[i].position, sorted_pop[i].position, opt->dim * sizeof(double));
        opt->population[i].fitness = sorted_pop[i].fitness;
    }

    free(temp_position);
    free(y);
    free(z);
    free(fitness);
    free(indices);
    for (int i = 0; i < opt->population_size; i++) {
        free(sorted_pop[i].position);
    }
    free(sorted_pop);
}

// Exploitation Phase (Local Search)
void puma_exploitation_phase(Optimizer *opt, PumaOptimizerState *state, int iter) {
    double T = (double)opt->max_iter;
    double *beta2 = (double *)malloc(opt->dim * sizeof(double));
    double *w = (double *)malloc(opt->dim * sizeof(double));
    double *v = (double *)malloc(opt->dim * sizeof(double));
    double *F1 = (double *)malloc(opt->dim * sizeof(double));
    double *F2 = (double *)malloc(opt->dim * sizeof(double));
    double *S1 = (double *)malloc(opt->dim * sizeof(double));
    double *S2 = (double *)malloc(opt->dim * sizeof(double));
    double *VEC = (double *)malloc(opt->dim * sizeof(double));
    double *Xatack = (double *)malloc(opt->dim * sizeof(double));
    double *mbest = (double *)malloc(opt->dim * sizeof(double));

    // Compute mean position (mbest)
    for (int j = 0; j < opt->dim; j++) {
        mbest[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            mbest[j] += opt->population[i].position[j];
        }
        mbest[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double beta1 = 2.0 * rand_double(0.0, 1.0);
        for (int j = 0; j < opt->dim; j++) {
            beta2[j] = rand_double(-1.0, 1.0); // Approximation of np.random.randn
            w[j] = rand_double(-1.0, 1.0);    // Eq 37
            v[j] = rand_double(-1.0, 1.0);    // Eq 38
            F1[j] = rand_double(-1.0, 1.0) * exp(2.0 - iter * (2.0 / T)); // Eq 35
        }
        double rand_val = rand_double(0.0, 1.0);
        for (int j = 0; j < opt->dim; j++) {
            F2[j] = w[j] * v[j] * v[j] * cos(2.0 * rand_val * w[j]); // Eq 36
            S1[j] = 2.0 * rand_double(0.0, 1.0) - 1.0 + rand_double(-1.0, 1.0);
            S2[j] = F1[j] * (2.0 * rand_double(0.0, 1.0) - 1.0) * opt->population[i].position[j] +
                    F2[j] * (1.0 - (2.0 * rand_double(0.0, 1.0) - 1.0)) * opt->best_solution.position[j];
            VEC[j] = S2[j] / S1[j];
        }

        if (rand_val <= 0.5) {
            memcpy(Xatack, VEC, opt->dim * sizeof(double));
            if (rand_double(0.0, 1.0) > state->Q) {
                int r1 = rand() % opt->population_size;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = opt->best_solution.position[j] +
                                                    beta1 * exp(beta2[j]) * (opt->population[r1].position[j] - opt->population[i].position[j]); // Eq 32
                }
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = beta1 * Xatack[j] - opt->best_solution.position[j]; // Eq 32
                }
            }
        } else {
            int r1 = 1 + (rand() % (opt->population_size - 1));
            double sign = (rand() % 2) ? 1.0 : -1.0;
            double denom = 1.0 + (state->beta * rand_double(0.0, 1.0));
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = (mbest[j] * opt->population[r1].position[j] - sign * opt->population[i].position[j]) / denom; // Eq 32
            }
        }
    }

    enforce_puma_bounds(opt);
    free(beta2);
    free(w);
    free(v);
    free(F1);
    free(F2);
    free(S1);
    free(S2);
    free(VEC);
    free(Xatack);
    free(mbest);
}

// Main Optimization Function
void PuO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (opt->dim <= 0 || opt->population_size <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Error: Invalid optimizer parameters\n");
        return;
    }

    PumaOptimizerState state;
    state.Q = Q_PROBABILITY;
    state.beta = BETA_FACTOR;
    state.PF[0] = PF1;
    state.PF[1] = PF2;
    state.PF[2] = PF3;
    state.mega_explore = MEGA_EXPLORE_INIT;
    state.mega_exploit = MEGA_EXPLOIT_INIT;
    state.unselected[0] = 1.0;
    state.unselected[1] = 1.0;
    state.f3_explore = 0.0;
    state.f3_exploit = 0.0;
    for (int i = 0; i < 3; i++) {
        state.seq_time_explore[i] = 1.0;
        state.seq_time_exploit[i] = 1.0;
        state.seq_cost_explore[i] = 1.0;
        state.seq_cost_exploit[i] = 1.0;
    }
    state.score_explore = 0.0;
    state.score_exploit = 0.0;
    state.pf_f3 = (double *)malloc(PF_F3_MAX_SIZE * sizeof(double)); // Max 6 entries
    state.pf_f3_size = 0;
    state.flag_change = 1;

    printf("Allocated state.pf_f3 with size %d\n", PF_F3_MAX_SIZE);

    initialize_pumas(opt);
    evaluate_pumas(opt, objective_function);
    int min_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[min_idx].fitness) {
            min_idx = i;
        }
    }
    opt->best_solution.fitness = opt->population[min_idx].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[min_idx].position[j];
    }
    double initial_best_cost = opt->best_solution.fitness;

    double *costs_explore = (double *)malloc(3 * sizeof(double));
    double *costs_exploit = (double *)malloc(3 * sizeof(double));

    // Unexperienced Phase (first 3 iterations)
    for (int iter = 0; iter < 3; iter++) {
        puma_exploration_phase(opt, &state, objective_function);
        evaluate_pumas(opt, objective_function);
        double min_fitness = opt->population[0].fitness;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
            }
        }
        costs_explore[iter] = min_fitness;

        puma_exploitation_phase(opt, &state, iter + 1);
        evaluate_pumas(opt, objective_function);
        min_fitness = opt->population[0].fitness;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
            }
        }
        costs_exploit[iter] = min_fitness;

        // Combine populations (simplified: keep current population)
        min_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[min_idx].fitness) {
                min_idx = i;
            }
        }
        opt->best_solution.fitness = objective_function(opt->population[min_idx].position);
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[min_idx].position[j];
        }
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Hyper Initialization
    state.seq_cost_explore[0] = fabs(initial_best_cost - costs_explore[0]); // Eq 5
    state.seq_cost_exploit[0] = fabs(initial_best_cost - costs_exploit[0]); // Eq 8
    state.seq_cost_explore[1] = fabs(costs_explore[1] - costs_explore[0]);  // Eq 6
    state.seq_cost_exploit[1] = fabs(costs_exploit[1] - costs_exploit[0]);  // Eq 9
    state.seq_cost_explore[2] = fabs(costs_explore[2] - costs_explore[1]);  // Eq 7
    state.seq_cost_exploit[2] = fabs(costs_exploit[2] - costs_exploit[1]); // Eq 10

    printf("Before hyper init: pf_f3_size = %d\n", state.pf_f3_size);
    for (int i = 0; i < 3; i++) {
        if (state.seq_cost_explore[i] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.seq_cost_explore[i];
        }
        if (state.seq_cost_exploit[i] != 0 && state.pf_f3_size < PF_F3_MAX_SIZE) {
            state.pf_f3[state.pf_f3_size++] = state.seq_cost_exploit[i];
        }
    }
    printf("After hyper init: pf_f3_size = %d\n", state.pf_f3_size);

    if (state.pf_f3_size == 0) {
        state.pf_f3[state.pf_f3_size++] = 1e-10;
    }

    double f1_explore = state.PF[0] * (state.seq_cost_explore[0] / state.seq_time_explore[0]); // Eq 1
    double f1_exploit = state.PF[0] * (state.seq_cost_exploit[0] / state.seq_time_exploit[0]); // Eq 2
    double sum_cost_explore = state.seq_cost_explore[0] + state.seq_cost_explore[1] + state.seq_cost_explore[2];
    double sum_cost_exploit = state.seq_cost_exploit[0] + state.seq_cost_exploit[1] + state.seq_cost_exploit[2];
    double sum_time_explore = state.seq_time_explore[0] + state.seq_time_explore[1] + state.seq_time_explore[2];
    double sum_time_exploit = state.seq_time_exploit[0] + state.seq_time_exploit[1] + state.seq_time_exploit[2];
    double f2_explore = state.PF[1] * (sum_cost_explore / sum_time_explore); // Eq 3
    double f2_exploit = state.PF[1] * (sum_cost_exploit / sum_time_exploit); // Eq 4
    state.score_explore = (state.PF[0] * f1_explore) + (state.PF[1] * f2_explore); // Eq 11
    state.score_exploit = (state.PF[0] * f1_exploit) + (state.PF[1] * f2_exploit); // Eq 12

    // Experienced Phase
    for (int iter = 3; iter < opt->max_iter; iter++) {
        int select_flag;
        double t_best_cost;
        double *t_best = (double *)malloc(opt->dim * sizeof(double));

        printf("Iteration %d: Before phase, pf_f3_size = %d\n", iter + 1, state.pf_f3_size);

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

        printf("Iteration %d: After phase, pf_f3_size = %d\n", iter + 1, state.pf_f3_size);

        if (state.flag_change != select_flag) {
            state.flag_change = select_flag;
            state.seq_time_explore[2] = state.seq_time_explore[1];
            state.seq_time_explore[1] = state.seq_time_explore[0];
            state.seq_time_explore[0] = state.unselected[0];
            state.seq_time_exploit[2] = state.seq_time_exploit[1];
            state.seq_time_exploit[1] = state.seq_time_exploit[0];
            state.seq_time_exploit[0] = state.unselected[1];
        }

        // Update scores
        double f1_explore = state.PF[0] * (state.seq_cost_explore[0] / state.seq_time_explore[0]); // Eq 14
        double f1_exploit = state.PF[0] * (state.seq_cost_exploit[0] / state.seq_time_exploit[0]); // Eq 13
        double f2_explore = state.PF[1] * (sum_cost_explore / sum_time_explore); // Eq 16
        double f2_exploit = state.PF[1] * (sum_cost_exploit / sum_time_exploit); // Eq 15

        if (state.score_explore < state.score_exploit) {
            state.mega_explore = state.mega_explore > 0.01 ? state.mega_explore - 0.01 : 0.01;
            state.mega_exploit = MEGA_EXPLOIT_INIT;
        } else {
            state.mega_explore = MEGA_EXPLORE_INIT;
            state.mega_exploit = state.mega_exploit > 0.01 ? state.mega_exploit - 0.01 : 0.01;
        }

        double lmn_explore = 1.0 - state.mega_explore; // Eq 24
        double lmn_exploit = 1.0 - state.mega_exploit; // Eq 22
        double min_pf_f3 = state.pf_f3[0];
        for (int i = 1; i < state.pf_f3_size; i++) {
            if (state.pf_f3[i] < min_pf_f3) min_pf_f3 = state.pf_f3[i];
        }
        state.score_explore = state.mega_explore * f1_explore + state.mega_explore * f2_explore + lmn_explore * (min_pf_f3 * state.f3_explore); // Eq 20
        state.score_exploit = state.mega_exploit * f1_exploit + state.mega_exploit * f2_exploit + lmn_exploit * (min_pf_f3 * state.f3_exploit); // Eq 19

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
        free(t_best);
    }

    printf("Freeing state.pf_f3, size = %d\n", state.pf_f3_size);
    free(state.pf_f3);
    free(costs_explore);
    free(costs_exploit);
}
