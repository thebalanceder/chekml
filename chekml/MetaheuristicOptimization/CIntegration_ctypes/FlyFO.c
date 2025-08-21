/* FlyFO.c - Implementation file for Flying Fox Optimization (FFO) */
#include "FlyFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed the random generator
#include <string.h>  // For memcpy()

// 🌟 Function to generate a random double between min and max
double rand_double(double min, double max);

// 🦇 Initialize Population
void initialize_population_flyfo(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
        ctx->past_fitness[i] = opt->population[i].fitness;
    }
    enforce_bound_constraints(opt);

    // Find best and worst solutions
    double best_fitness = INFINITY;
    double worst_fitness = -INFINITY;
    int best_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < best_fitness) {
            best_fitness = opt->population[i].fitness;
            best_idx = i;
        }
        if (opt->population[i].fitness > worst_fitness) {
            worst_fitness = opt->population[i].fitness;
        }
    }
    opt->best_solution.fitness = best_fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[best_idx].position[j];
    }
    ctx->worst_fitness = worst_fitness;
    printf("Population initialized\n");
}

// 🦇 Fuzzy Self-Tuning for Alpha and Pa Parameters
void fuzzy_self_tuning(Optimizer *opt, FlyFOContext *ctx, int i, double *alpha, double *pa) {
    double delta = fabs(opt->best_solution.fitness - opt->population[i].fitness);
    double deltamax = fabs(opt->best_solution.fitness - ctx->worst_fitness);
    double fi = (opt->population[i].fitness - ctx->past_fitness[i]) / deltamax;
    if (deltamax == 0.0) fi = 0.0;

    double deltas[3] = {DELTASO_0 * deltamax, DELTASO_1 * deltamax, DELTASO_2 * deltamax};
    double alpha_params[3] = {ALPHA_PARAM_0, ALPHA_PARAM_1, ALPHA_PARAM_2};
    double pa_params[3] = {PA_PARAM_0, PA_PARAM_1, PA_PARAM_2};

    // Delta membership
    double delta_membership[3] = {0.0, 0.0, 0.0}; // Same, Near, Far
    double fi_membership[3] = {0.0, 1.0 - fabs(fi), 0.0}; // Better, Same, Worse

    if (0 <= delta && delta < deltas[1]) {
        if (delta < deltas[0]) {
            delta_membership[0] = 1.0; // Same
        } else {
            delta_membership[0] = (deltas[1] - delta) / (deltas[1] - deltas[0]); // Same
            delta_membership[1] = (delta - deltas[0]) / (deltas[1] - deltas[0]); // Near
        }
    } else if (deltas[1] <= delta && delta <= deltamax) {
        if (deltas[1] <= delta && delta <= deltas[2]) {
            delta_membership[1] = (deltas[2] - delta) / (deltas[2] - deltas[1]); // Near
            delta_membership[2] = (delta - deltas[1]) / (deltas[2] - deltas[1]); // Far
        } else {
            delta_membership[2] = 1.0; // Far
        }
    }

    if (-1.0 <= fi && fi <= 1.0) {
        if (-1.0 <= fi && fi < 0.0) {
            fi_membership[0] = -fi; // Better
            fi_membership[1] = 0.0; // Same
        } else if (0.0 < fi && fi <= 1.0) {
            fi_membership[2] = fi; // Worse
            fi_membership[1] = 0.0; // Same
        }
    }

    // Alpha rules
    double ruleno_alpha[3] = {0.0};
    ruleno_alpha[0] = fi_membership[0]; // Better
    ruleno_alpha[1] = fmax(fi_membership[1], fmax(delta_membership[0], delta_membership[1])); // Same or Near
    ruleno_alpha[2] = fmax(fi_membership[2], delta_membership[2]); // Worse or Far
    double alpha_sum = ruleno_alpha[0] + ruleno_alpha[1] + ruleno_alpha[2];
    *alpha = (alpha_sum > 0.0) ? (ruleno_alpha[0] * alpha_params[0] + 
                                  ruleno_alpha[1] * alpha_params[1] + 
                                  ruleno_alpha[2] * alpha_params[2]) / alpha_sum : 1.0;

    // Pa rules
    double ruleno_pa[3] = {0.0};
    ruleno_pa[0] = fmax(fi_membership[2], delta_membership[2]); // Worse or Far
    ruleno_pa[1] = fmax(fi_membership[1], delta_membership[0]); // Same
    ruleno_pa[2] = fmax(fi_membership[0], delta_membership[1]); // Better or Near
    double pa_sum = ruleno_pa[0] + ruleno_pa[1] + ruleno_pa[2];
    *pa = (pa_sum > 0.0) ? (ruleno_pa[0] * pa_params[0] + 
                            ruleno_pa[1] * pa_params[1] + 
                            ruleno_pa[2] * pa_params[2]) / pa_sum : 0.85;
}

// 🦇 Update Position (Global and Local Search)
void update_position(Optimizer *opt, FlyFOContext *ctx, int i, double alpha, double pa, double (*objective_function)(double *)) {
    double new_position[opt->dim];
    double new_fitness;
    double deltas[3] = {DELTASO_0 * fabs(opt->best_solution.fitness - ctx->worst_fitness),
                        DELTASO_1 * fabs(opt->best_solution.fitness - ctx->worst_fitness),
                        DELTASO_2 * fabs(opt->best_solution.fitness - ctx->worst_fitness)};

    if (fabs(opt->population[i].fitness - opt->best_solution.fitness) > (deltas[0] * 0.5)) {
        // Global movement towards best solution
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + 
                             alpha * rand_double(0.0, 1.0) * 
                             (opt->best_solution.position[j] - opt->population[i].position[j]);
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
            if (perm[k] != i) {
                A[a_idx++] = perm[k];
            }
        }
        double stepsize[opt->dim];
        for (int j = 0; j < opt->dim; j++) {
            stepsize[j] = rand_double(0.0, 1.0) * (opt->best_solution.position[j] - opt->population[i].position[j]) +
                          rand_double(0.0, 1.0) * (opt->population[A[0]].position[j] - opt->population[A[1]].position[j]);
        }
        memcpy(new_position, opt->population[i].position, opt->dim * sizeof(double));
        int j0 = rand() % opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            if (j == j0 || rand_double(0.0, 1.0) >= pa) {
                new_position[j] += stepsize[j];
            }
        }
    }

    // Enforce bounds and evaluate
    for (int j = 0; j < opt->dim; j++) {
        if (new_position[j] < opt->bounds[2 * j]) new_position[j] = opt->bounds[2 * j];
        if (new_position[j] > opt->bounds[2 * j + 1]) new_position[j] = opt->bounds[2 * j + 1];
    }
    new_fitness = objective_function(new_position);

    // Update population if better
    if (new_fitness < opt->population[i].fitness) {
        ctx->past_fitness[i] = opt->population[i].fitness;
        opt->population[i].fitness = new_fitness;
        memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(opt->best_solution.position, new_position, opt->dim * sizeof(double));
        }
    }
    if (new_fitness > ctx->worst_fitness) {
        ctx->worst_fitness = new_fitness;
    }

    // Update survival list if condition met
    if (fabs(new_fitness - opt->best_solution.fitness) > deltas[2]) {
        ctx->past_fitness[i] = opt->population[i].fitness;
        replace_with_survival_list(opt, ctx, opt->population[i].position);
        opt->population[i].fitness = objective_function(opt->population[i].position);
        ctx->past_fitness[i] = opt->population[i].fitness;
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
        if (opt->population[i].fitness > ctx->worst_fitness) {
            ctx->worst_fitness = opt->population[i].fitness;
        }
    }

    update_survival_list(opt, ctx, new_position, new_fitness);
}

// 🦇 Update Survival List
void update_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position, double new_fitness) {
    int surv_list_size = (int)(SURVIVAL_LIST_RATIO * opt->population_size);
    if (ctx->survival_list_count >= surv_list_size) return;

    // Allocate new entry
    if (ctx->survival_list == NULL) {
        ctx->survival_list = (Solution *)malloc(surv_list_size * sizeof(Solution));
        ctx->survival_list_count = 0;
    }
    if (ctx->survival_list_count == 0 || 
        (ctx->survival_list_count > 0 && new_fitness < ctx->survival_list[ctx->survival_list_count - 1].fitness)) {
        ctx->survival_list[ctx->survival_list_count].position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(ctx->survival_list[ctx->survival_list_count].position, new_position, opt->dim * sizeof(double));
        ctx->survival_list[ctx->survival_list_count].fitness = new_fitness;
        ctx->survival_list_count++;
    }

    // Sort survival list by fitness
    for (int i = 1; i < ctx->survival_list_count; i++) {
        Solution key = ctx->survival_list[i];
        int j = i - 1;
        while (j >= 0 && ctx->survival_list[j].fitness > key.fitness) {
            ctx->survival_list[j + 1] = ctx->survival_list[j];
            j--;
        }
        ctx->survival_list[j + 1] = key;
    }
    if (ctx->survival_list_count > surv_list_size) {
        free(ctx->survival_list[ctx->survival_list_count - 1].position);
        ctx->survival_list_count--;
    }
}

// 🦇 Replace with Survival List
void replace_with_survival_list(Optimizer *opt, FlyFOContext *ctx, double *new_position) {
    if (ctx->survival_list_count < 2) {
        for (int j = 0; j < opt->dim; j++) {
            new_position[j] = opt->bounds[2 * j] + 
                             rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        return;
    }

    int m = rand() % (ctx->survival_list_count - 1) + 2; // Random number between 2 and survival_list_count
    int indices[m];
    for (int k = 0; k < m; k++) {
        indices[k] = rand() % ctx->survival_list_count;
        for (int l = 0; l < k; l++) {
            if (indices[k] == indices[l]) {
                k--;
                break;
            }
        }
    }

    for (int j = 0; j < opt->dim; j++) {
        new_position[j] = 0.0;
        for (int k = 0; k < m; k++) {
            new_position[j] += ctx->survival_list[indices[k]].position[j];
        }
        new_position[j] /= m;
        if (new_position[j] < opt->bounds[2 * j]) new_position[j] = opt->bounds[2 * j];
        if (new_position[j] > opt->bounds[2 * j + 1]) new_position[j] = opt->bounds[2 * j + 1];
    }
}

// 🦇 Crossover between Two Parents
void crossover(Optimizer *opt, int parent1, int parent2, double *off1, double *off2) {
    double extracros = 0.0;
    for (int j = 0; j < opt->dim; j++) {
        double L = rand_double(-extracros, 1.0 + extracros);
        off1[j] = L * opt->population[parent1].position[j] + (1.0 - L) * opt->population[parent2].position[j];
        off2[j] = L * opt->population[parent2].position[j] + (1.0 - L) * opt->population[parent1].position[j];
        if (off1[j] < opt->bounds[2 * j]) off1[j] = opt->bounds[2 * j];
        if (off1[j] > opt->bounds[2 * j + 1]) off1[j] = opt->bounds[2 * j + 1];
        if (off2[j] < opt->bounds[2 * j]) off2[j] = opt->bounds[2 * j];
        if (off2[j] > opt->bounds[2 * j + 1]) off2[j] = opt->bounds[2 * j + 1];
    }
}

// 🦇 Suffocation Phase (Replace Redundant Best Solutions)
void suffocation_phase(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *)) {
    int best_count = 0;
    int best_indices[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness == opt->best_solution.fitness) {
            best_indices[best_count++] = i;
        }
    }
    double p_death = (best_count > 1) ? (double)(best_count - 1) / opt->population_size : 0.0;

    for (int i = 0; i < best_count; i += 2) {
        if (rand_double(0.0, 1.0) < p_death) {
            if (i == best_count - 1 && best_count % 2 == 1) {
                replace_with_survival_list(opt, ctx, opt->population[best_indices[i]].position);
                opt->population[best_indices[i]].fitness = objective_function(opt->population[best_indices[i]].position);
                ctx->past_fitness[best_indices[i]] = opt->population[best_indices[i]].fitness;
                update_survival_list(opt, ctx, opt->population[best_indices[i]].position, opt->population[best_indices[i]].fitness);
            } else {
                int parent1 = rand() % opt->population_size;
                int parent2 = rand() % opt->population_size;
                double off1[opt->dim], off2[opt->dim];
                if (rand_double(0.0, 1.0) < 0.5 && 
                    opt->population[parent1].fitness != opt->population[parent2].fitness) {
                    crossover(opt, parent1, parent2, off1, off2);
                } else {
                    replace_with_survival_list(opt, ctx, off1);
                    replace_with_survival_list(opt, ctx, off2);
                }

                memcpy(opt->population[best_indices[i]].position, off1, opt->dim * sizeof(double));
                opt->population[best_indices[i]].fitness = objective_function(off1);
                ctx->past_fitness[best_indices[i]] = opt->population[best_indices[i]].fitness;
                update_survival_list(opt, ctx, off1, opt->population[best_indices[i]].fitness);

                if (i + 1 < best_count) {
                    memcpy(opt->population[best_indices[i + 1]].position, off2, opt->dim * sizeof(double));
                    opt->population[best_indices[i + 1]].fitness = objective_function(off2);
                    ctx->past_fitness[best_indices[i + 1]] = opt->population[best_indices[i + 1]].fitness;
                    update_survival_list(opt, ctx, off2, opt->population[best_indices[i + 1]].fitness);
                }

                for (int idx = i; idx < i + 2 && idx < best_count; idx++) {
                    if (opt->population[best_indices[idx]].fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = opt->population[best_indices[idx]].fitness;
                        memcpy(opt->best_solution.position, opt->population[best_indices[idx]].position, 
                               opt->dim * sizeof(double));
                    }
                    if (opt->population[best_indices[idx]].fitness > ctx->worst_fitness) {
                        ctx->worst_fitness = opt->population[best_indices[idx]].fitness;
                    }
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
    FlyFOContext ctx;
    ctx.worst_fitness = -INFINITY;
    ctx.survival_list = NULL;
    ctx.survival_list_count = 0;
    ctx.past_fitness = (double *)malloc(opt->population_size * sizeof(double));

    initialize_population_flyfo(opt, &ctx, objective_function);
    int func_count = opt->population_size;
    int iteration = 0;

    double deltasO[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    double deltasO_max[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    double deltasO_min[3] = {DELTASO_0 / 10, DELTASO_1 / 10, DELTASO_2 / 10};

    while (func_count < MAX_EVALS_DEFAULT) {
        iteration++;
        printf("Starting iteration %d\n", iteration);

        for (int i = 0; i < opt->population_size; i++) {
            printf("Processing fox %d\n", i);
            double alpha, pa;
            fuzzy_self_tuning(opt, &ctx, i, &alpha, &pa);
            update_position(opt, &ctx, i, alpha, pa, objective_function);
            func_count += 2; // One for update_position, one for potential survival list replacement
        }

        int best_count = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == opt->best_solution.fitness) {
                best_count++;
            }
        }
        suffocation_phase(opt, &ctx, objective_function);
        func_count += best_count; // Approximate additional evaluations in suffocation

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
