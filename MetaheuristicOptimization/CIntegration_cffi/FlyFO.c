/* FlyFO.c - Optimized Implementation file for Flying Fox Optimization (FFO) */
#include "FlyFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy()

// 🌟 Inline function to generate a random double between min and max
double rand_double(double min, double max);

// 🦇 Initialize Population
void initialize_population_flyfo(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *)) {
    double *bounds = opt->bounds;
    Solution *pop = opt->population;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    double best_fitness = INFINITY;
    double worst_fitness = -INFINITY;
    int best_idx = 0;

    for (int i = 0; i < pop_size; i++) {
        double *pos = pop[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] = bounds[2 * j] + rand_double(0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        pop[i].fitness = objective_function(pos);
        ctx->past_fitness[i] = pop[i].fitness;

        if (pop[i].fitness < best_fitness) {
            best_fitness = pop[i].fitness;
            best_idx = i;
        }
        if (pop[i].fitness > worst_fitness) {
            worst_fitness = pop[i].fitness;
        }
    }

    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, pop[best_idx].position, dim * sizeof(double));
    ctx->worst_fitness = worst_fitness;
    printf("Population initialized\n");
}

// 🦇 Fuzzy Self-Tuning for Alpha and Pa Parameters
void fuzzy_self_tuning(Optimizer *opt, FlyFOContext *ctx, int i, double deltamax, double *alpha, double *pa) {
    double delta = fabs(opt->best_solution.fitness - opt->population[i].fitness);
    double fi = deltamax > 0.0 ? (opt->population[i].fitness - ctx->past_fitness[i]) / deltamax : 0.0;

    double deltas[3] = {DELTASO_0 * deltamax, DELTASO_1 * deltamax, DELTASO_2 * deltamax};
    static const double alpha_params[3] = {ALPHA_PARAM_0, ALPHA_PARAM_1, ALPHA_PARAM_2};
    static const double pa_params[3] = {PA_PARAM_0, PA_PARAM_1, PA_PARAM_2};

    // Delta membership
    double delta_membership[3] = {0.0};
    double fi_membership[3] = {0.0, 1.0 - fabs(fi), 0.0}; // Better, Same, Worse

    if (delta < deltas[1]) {
        if (delta < deltas[0]) {
            delta_membership[0] = 1.0; // Same
        } else {
            delta_membership[0] = (deltas[1] - delta) / (deltas[1] - deltas[0]); // Same
            delta_membership[1] = (delta - deltas[0]) / (deltas[1] - deltas[0]); // Near
        }
    } else if (delta <= deltamax) {
        if (delta <= deltas[2]) {
            delta_membership[1] = (deltas[2] - delta) / (deltas[2] - deltas[1]); // Near
            delta_membership[2] = (delta - deltas[1]) / (deltas[2] - deltas[1]); // Far
        } else {
            delta_membership[2] = 1.0; // Far
        }
    }

    if (-1.0 <= fi && fi <= 1.0) {
        if (fi < 0.0) {
            fi_membership[0] = -fi; // Better
            fi_membership[1] = 0.0;
        } else if (fi > 0.0) {
            fi_membership[2] = fi; // Worse
            fi_membership[1] = 0.0;
        }
    }

    // Alpha rules
    double ruleno_alpha[3];
    ruleno_alpha[0] = fi_membership[0];
    ruleno_alpha[1] = fmax(fi_membership[1], fmax(delta_membership[0], delta_membership[1]));
    ruleno_alpha[2] = fmax(fi_membership[2], delta_membership[2]);
    double alpha_sum = ruleno_alpha[0] + ruleno_alpha[1] + ruleno_alpha[2];
    *alpha = alpha_sum > 0.0 ? (ruleno_alpha[0] * alpha_params[0] + 
                                ruleno_alpha[1] * alpha_params[1] + 
                                ruleno_alpha[2] * alpha_params[2]) / alpha_sum : 1.0;

    // Pa rules
    double ruleno_pa[3];
    ruleno_pa[0] = fmax(fi_membership[2], delta_membership[2]);
    ruleno_pa[1] = fmax(fi_membership[1], delta_membership[0]);
    ruleno_pa[2] = fmax(fi_membership[0], delta_membership[1]);
    double pa_sum = ruleno_pa[0] + ruleno_pa[1] + ruleno_pa[2];
    *pa = pa_sum > 0.0 ? (ruleno_pa[0] * pa_params[0] + 
                          ruleno_pa[1] * pa_params[1] + 
                          ruleno_pa[2] * pa_params[2]) / pa_sum : 0.85;
}

// 🦇 Update Position (Global and Local Search)
void update_position(Optimizer *opt, FlyFOContext *ctx, int i, double alpha, double pa, double *deltas, double (*objective_function)(double *)) {
    int dim = opt->dim;
    double *pos = opt->population[i].position;
    double *best_pos = opt->best_solution.position;
    double *bounds = opt->bounds;
    double new_fitness;

    if (fabs(opt->population[i].fitness - opt->best_solution.fitness) > (deltas[0] * 0.5)) {
        // Global movement towards best solution
        for (int j = 0; j < dim; j++) {
            pos[j] += alpha * rand_double(0.0, 1.0) * (best_pos[j] - pos[j]);
            if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
            if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
        }
    } else {
        // Local search with random steps
        int A[2];
        int perm[opt->population_size];
        for (int k = 0; k < opt->population_size; k++) perm[k] = k;
        for (int k = opt->population_size - 1; k > 0; k--) {
            int r = rand() % (k + 1);
            int temp = perm[k];
            perm[k] = perm[r];
            perm[r] = temp;
        }
        int a_idx = 0;
        for (int k = 0; k < opt->population_size && a_idx < 2; k++) {
            if (perm[k] != i) A[a_idx++] = perm[k];
        }

        for (int j = 0; j < dim; j++) {
            double step = rand_double(0.0, 1.0) * (best_pos[j] - pos[j]) +
                          rand_double(0.0, 1.0) * (opt->population[A[0]].position[j] - opt->population[A[1]].position[j]);
            if (j == (rand() % dim) || rand_double(0.0, 1.0) >= pa) {
                pos[j] += step;
                if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
                if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
            }
        }
    }

    new_fitness = objective_function(pos);

    // Update population if better
    if (new_fitness < opt->population[i].fitness) {
        ctx->past_fitness[i] = opt->population[i].fitness;
        opt->population[i].fitness = new_fitness;
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(opt->best_solution.position, pos, dim * sizeof(double));
        }
    }
    if (new_fitness > ctx->worst_fitness) {
        ctx->worst_fitness = new_fitness;
    }

    // Update survival list if condition met
    if (fabs(new_fitness - opt->best_solution.fitness) > deltas[2]) {
        ctx->past_fitness[i] = opt->population[i].fitness;
        replace_with_survival_list(opt, ctx, pos, opt->population_size * SURVIVAL_LIST_RATIO);
        opt->population[i].fitness = objective_function(pos);
        ctx->past_fitness[i] = opt->population[i].fitness;
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, pos, dim * sizeof(double));
        }
        if (opt->population[i].fitness > ctx->worst_fitness) {
            ctx->worst_fitness = opt->population[i].fitness;
        }
        update_survival_list(opt, ctx, pos, opt->population[i].fitness, opt->population_size * SURVIVAL_LIST_RATIO);
    } else {
        update_survival_list(opt, ctx, pos, new_fitness, opt->population_size * SURVIVAL_LIST_RATIO);
    }
}

// 🦇 Update Survival List
void update_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position, double new_fitness, int surv_list_size) {
    if (ctx->survival_list_count >= surv_list_size) return;

    if (!ctx->survival_list) {
        ctx->survival_list = (Solution *)malloc(surv_list_size * sizeof(Solution));
        ctx->survival_list_count = 0;
    }

    if (ctx->survival_list_count == 0 || 
        (ctx->survival_list_count > 0 && new_fitness < ctx->survival_list[ctx->survival_list_count - 1].fitness)) {
        Solution *entry = &ctx->survival_list[ctx->survival_list_count];
        entry->position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(entry->position, new_position, opt->dim * sizeof(double));
        entry->fitness = new_fitness;
        ctx->survival_list_count++;
    }

    // Insertion sort for small lists
    for (int i = ctx->survival_list_count - 1; i > 0; i--) {
        if (ctx->survival_list[i].fitness < ctx->survival_list[i - 1].fitness) {
            Solution temp = ctx->survival_list[i];
            ctx->survival_list[i] = ctx->survival_list[i - 1];
            ctx->survival_list[i - 1] = temp;
        } else {
            break;
        }
    }

    if (ctx->survival_list_count > surv_list_size) {
        free(ctx->survival_list[ctx->survival_list_count - 1].position);
        ctx->survival_list_count--;
    }
}

// 🦇 Replace with Survival List
void replace_with_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position, int surv_list_size) {
    if (ctx->survival_list_count < 2) {
        double *bounds = opt->bounds;
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = bounds[2 * j] + rand_double(0.0, 1.0) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        return;
    }

    int m = rand() % (ctx->survival_list_count - 1) + 2;
    int indices[10]; // Stack allocation for small m
    if (m > 10) {
        indices[0] = rand() % ctx->survival_list_count; // Fallback for large m
        m = 1;
    } else {
        for (int k = 0; k < m; k++) {
            indices[k] = rand() % ctx->survival_list_count;
            for (int l = 0; l < k; l++) {
                if (indices[k] == indices[l]) {
                    k--;
                    break;
                }
            }
        }
    }

    double *bounds = opt->bounds;
    for (int j = 0; j < opt->dim; j++) {
        new_position[j] = 0.0;
        for (int k = 0; k < m; k++) {
            new_position[j] += ctx->survival_list[indices[k]].position[j];
        }
        new_position[j] /= m;
        if (new_position[j] < bounds[2 * j]) new_position[j] = bounds[2 * j];
        if (new_position[j] > bounds[2 * j + 1]) new_position[j] = bounds[2 * j + 1];
    }
}

// 🦇 Crossover between Two Parents
void crossover(Optimizer *opt, int parent1, int parent2, double *off1, double *off2, double *randoms, int dim) {
    double *p1 = opt->population[parent1].position;
    double *p2 = opt->population[parent2].position;
    double *bounds = opt->bounds;

    for (int j = 0; j < dim; j++) {
        double L = randoms[j];
        off1[j] = L * p1[j] + (1.0 - L) * p2[j];
        off2[j] = L * p2[j] + (1.0 - L) * p1[j];
        if (off1[j] < bounds[2 * j]) off1[j] = bounds[2 * j];
        if (off1[j] > bounds[2 * j + 1]) off1[j] = bounds[2 * j + 1];
        if (off2[j] < bounds[2 * j]) off2[j] = bounds[2 * j];
        if (off2[j] > bounds[2 * j + 1]) off2[j] = bounds[2 * j + 1];
    }
}

// 🦇 Suffocation Phase (Replace Redundant Best Solutions)
void suffocation_phase(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *), int surv_list_size) {
    int best_count = 0;
    int best_indices[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness == opt->best_solution.fitness) {
            best_indices[best_count++] = i;
        }
    }
    double p_death = best_count > 1 ? (double)(best_count - 1) / opt->population_size : 0.0;

    for (int i = 0; i < best_count; i += 2) {
        if (rand_double(0.0, 1.0) >= p_death) continue;

        if (i == best_count - 1 && best_count % 2 == 1) {
            double *pos = opt->population[best_indices[i]].position;
            replace_with_survival_list(opt, ctx, pos, surv_list_size);
            opt->population[best_indices[i]].fitness = objective_function(pos);
            ctx->past_fitness[best_indices[i]] = opt->population[best_indices[i]].fitness;
            update_survival_list(opt, ctx, pos, opt->population[best_indices[i]].fitness, surv_list_size);
        } else {
            int parent1 = rand() % opt->population_size;
            int parent2 = rand() % opt->population_size;
            double off1[opt->dim], off2[opt->dim];
            double randoms[opt->dim];
            for (int j = 0; j < opt->dim; j++) randoms[j] = rand_double(0.0, 1.0);

            if (rand_double(0.0, 1.0) < 0.5 && 
                opt->population[parent1].fitness != opt->population[parent2].fitness) {
                crossover(opt, parent1, parent2, off1, off2, randoms, opt->dim);
            } else {
                replace_with_survival_list(opt, ctx, off1, surv_list_size);
                replace_with_survival_list(opt, ctx, off2, surv_list_size);
            }

            double *pos1 = opt->population[best_indices[i]].position;
            memcpy(pos1, off1, opt->dim * sizeof(double));
            opt->population[best_indices[i]].fitness = objective_function(pos1);
            ctx->past_fitness[best_indices[i]] = opt->population[best_indices[i]].fitness;
            update_survival_list(opt, ctx, pos1, opt->population[best_indices[i]].fitness, surv_list_size);

            if (i + 1 < best_count) {
                double *pos2 = opt->population[best_indices[i + 1]].position;
                memcpy(pos2, off2, opt->dim * sizeof(double));
                opt->population[best_indices[i + 1]].fitness = objective_function(pos2);
                ctx->past_fitness[best_indices[i + 1]] = opt->population[best_indices[i + 1]].fitness;
                update_survival_list(opt, ctx, pos2, opt->population[best_indices[i + 1]].fitness, surv_list_size);
            }

            for (int idx = i; idx < i + 2 && idx < best_count; idx++) {
                Solution *sol = &opt->population[best_indices[idx]];
                if (sol->fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = sol->fitness;
                    memcpy(opt->best_solution.position, sol->position, opt->dim * sizeof(double));
                }
                if (sol->fitness > ctx->worst_fitness) {
                    ctx->worst_fitness = sol->fitness;
                }
            }
        }
    }
}

// 🚀 Main Optimization Function
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL)); // Seed random number generator
    printf("Starting optimization...\n");

    // Initialize FlyFO context
    FlyFOContext ctx = { .worst_fitness = -INFINITY, .survival_list = NULL, .survival_list_count = 0 };
    ctx.past_fitness = (double *)malloc(opt->population_size * sizeof(double));

    initialize_population_flyfo(opt, &ctx, objective_function);
    int func_count = opt->population_size;
    int iteration = 0;
    int surv_list_size = (int)(opt->population_size * SURVIVAL_LIST_RATIO);

    double deltasO[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    double deltasO_max[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    double deltasO_min[3] = {DELTASO_0 / 10, DELTASO_1 / 10, DELTASO_2 / 10};

    while (func_count < MAX_EVALS_DEFAULT) {
        iteration++;
        printf("Starting iteration %d\n", iteration);

        double deltamax = fabs(opt->best_solution.fitness - ctx.worst_fitness);
        double deltas[3] = {DELTASO_0 * deltamax, DELTASO_1 * deltamax, DELTASO_2 * deltamax};

        for (int i = 0; i < opt->population_size; i++) {
            double alpha, pa;
            fuzzy_self_tuning(opt, &ctx, i, deltamax, &alpha, &pa);
            update_position(opt, &ctx, i, alpha, pa, deltas, objective_function);
            func_count += 2; // One for update_position, one for potential survival list replacement
        }

        int best_count = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == opt->best_solution.fitness) best_count++;
        }
        suffocation_phase(opt, &ctx, objective_function, surv_list_size);
        func_count += best_count;

        // Update deltasO
        for (int k = 0; k < 3; k++) {
            deltasO[k] = deltasO_max[k] - ((deltasO_max[k] - deltasO_min[k]) / MAX_EVALS_DEFAULT) * func_count;
        }

        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", 
               iteration, func_count, opt->best_solution.fitness);
    }

    // Free survival list memory
    for (int i = 0; i < ctx.survival_list_count; i++) {
        free(ctx.survival_list[i].position);
    }
    free(ctx.survival_list);
    free(ctx.past_fitness);

    printf("Optimization completed\n");
}
