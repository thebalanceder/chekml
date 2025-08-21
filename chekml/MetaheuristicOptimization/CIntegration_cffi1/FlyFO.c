/* FlyFO.c - Ultra-Optimized Implementation for Flying Fox Optimization (FFO) */
#include "FlyFO.h"
#include <string.h>
#include <time.h>

// 🦇 Initialize Population
void initialize_population_flyfo(Optimizer *opt) {
    // Placeholder: Actual initialization moved to FlyFO_optimize
    // Generaloptimizer.c may set up population and bounds
    printf("FlyFO: Population initialization placeholder\n");
}

// 🦇 Update Position with SIMD
void update_position(Optimizer *opt, FlyFOContext *ctx, int i, double alpha, double pa, const double *deltas, double (*objective_function)(double *), unsigned int *rng_state) {
    const int dim = opt->dim;
    double *pos = opt->population[i].position;
    const double *best_pos = opt->best_solution.position;
    const double *bounds = opt->bounds;
    double fitness = opt->population[i].fitness;

    if (fabs(fitness - opt->best_solution.fitness) > deltas[0] * 0.5) {
        int j = 0;
        if (dim >= 4) {
            __m256d alpha_vec = _mm256_set1_pd(alpha);
            __m256d *p = (__m256d *)pos;
            const __m256d *bp = (const __m256d *)best_pos;
            for (; j <= dim - 4; j += 4, p++, bp++) {
                __m256d r = _mm256_set1_pd(fast_rand(0.0, 1.0, rng_state));
                __m256d diff = _mm256_sub_pd(*bp, *p);
                *p = _mm256_add_pd(*p, _mm256_mul_pd(_mm256_mul_pd(alpha_vec, r), diff));
            }
        }
        for (; j < dim; j++) {
            pos[j] += alpha * fast_rand(0.0, 1.0, rng_state) * (best_pos[j] - pos[j]);
        }
    } else {
        int A[2], perm[opt->population_size];
        for (int k = 0; k < opt->population_size; k++) perm[k] = k;
        for (int k = opt->population_size - 1; k > 0; k--) {
            int r = (int)fast_rand(0.0, k + 1, rng_state);
            int temp = perm[k];
            perm[k] = perm[r];
            perm[r] = temp;
        }
        int a_idx = 0;
        for (int k = 0; k < opt->population_size && a_idx < 2; k++) {
            if (perm[k] != i) A[a_idx++] = perm[k];
        }

        int j0 = (int)fast_rand(0.0, dim, rng_state);
        for (int j = 0; j < dim; j++) {
            double r1 = fast_rand(0.0, 1.0, rng_state), r2 = fast_rand(0.0, 1.0, rng_state);
            double step = r1 * (best_pos[j] - pos[j]) + r2 * (opt->population[A[0]].position[j] - opt->population[A[1]].position[j]);
            if (j == j0 || fast_rand(0.0, 1.0, rng_state) >= pa) pos[j] += step;
        }
    }

    enforce_bounds(pos, bounds, dim);
    double new_fitness = objective_function(pos);

    if (new_fitness < fitness) {
        ctx->past_fitness[i] = fitness;
        opt->population[i].fitness = new_fitness;
        if (new_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = new_fitness;
            memcpy(opt->best_solution.position, pos, dim * sizeof(double));
        }
    }
    if (new_fitness > ctx->worst_fitness) ctx->worst_fitness = new_fitness;

    if (fabs(new_fitness - opt->best_solution.fitness) > deltas[2]) {
        ctx->past_fitness[i] = opt->population[i].fitness;
        replace_with_survival_list(opt, ctx, pos, opt->population_size * SURVIVAL_LIST_RATIO, rng_state);
        opt->population[i].fitness = objective_function(pos);
        ctx->past_fitness[i] = opt->population[i].fitness;
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, pos, dim * sizeof(double));
        }
        if (opt->population[i].fitness > ctx->worst_fitness) ctx->worst_fitness = opt->population[i].fitness;
    }
    update_survival_list(opt, ctx, pos, new_fitness, opt->population_size * SURVIVAL_LIST_RATIO);
}

// 🦇 Update Survival List
void update_survival_list(Optimizer *opt, FlyFOContext *ctx, const double *pos, double fitness, int surv_list_size) {
    if (ctx->survival_list_count >= surv_list_size) return;

    if (!ctx->survival_list) {
        ctx->survival_list = (Solution *)malloc(surv_list_size * sizeof(Solution));
        ctx->survival_list_count = 0;
    }

    if (ctx->survival_list_count == 0 || fitness < ctx->survival_list[ctx->survival_list_count - 1].fitness) {
        Solution *entry = &ctx->survival_list[ctx->survival_list_count];
        entry->position = (double *)malloc(opt->dim * sizeof(double));
        memcpy(entry->position, pos, opt->dim * sizeof(double));
        entry->fitness = fitness;
        ctx->survival_list_count++;
    }

    for (int i = ctx->survival_list_count - 1; i > 0 && ctx->survival_list[i].fitness < ctx->survival_list[i - 1].fitness; i--) {
        Solution temp = ctx->survival_list[i];
        ctx->survival_list[i] = ctx->survival_list[i - 1];
        ctx->survival_list[i - 1] = temp;
    }

    if (ctx->survival_list_count > surv_list_size) {
        free(ctx->survival_list[ctx->survival_list_count - 1].position);
        ctx->survival_list_count--;
    }
}

// 🦇 Replace with Survival List
void replace_with_survival_list(Optimizer *opt, FlyFOContext *ctx, double *pos, int surv_list_size, unsigned int *rng_state) {
    if (ctx->survival_list_count < 2) {
        const double *bounds = opt->bounds;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = bounds[2 * j] + fast_rand(0.0, 1.0, rng_state) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        return;
    }

    int m = (int)fast_rand(0.0, ctx->survival_list_count - 1, rng_state) + 2;
    m = m > 8 ? 8 : m;
    int indices[8];
    for (int k = 0; k < m; k++) {
        indices[k] = (int)fast_rand(0.0, ctx->survival_list_count, rng_state);
        for (int l = 0; l < k; l++) {
            if (indices[k] == indices[l]) { k--; break; }
        }
    }

    const double *bounds = opt->bounds;
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) sum += ctx->survival_list[indices[k]].position[j];
        pos[j] = sum / m;
        if (pos[j] < bounds[2 * j]) pos[j] = bounds[2 * j];
        if (pos[j] > bounds[2 * j + 1]) pos[j] = bounds[2 * j + 1];
    }
}

// 🦇 Crossover with SIMD
void crossover(const Optimizer *opt, int p1, int p2, double *off1, double *off2, int dim, unsigned int *rng_state) {
    const double *p1_pos = opt->population[p1].position;
    const double *p2_pos = opt->population[p2].position;
    const double *bounds = opt->bounds;

    int i = 0;
    if (dim >= 4) {
        __m256d *o1 = (__m256d *)off1, *o2 = (__m256d *)off2;
        const __m256d *p1v = (const __m256d *)p1_pos, *p2v = (const __m256d *)p2_pos;
        for (; i <= dim - 4; i += 4, o1++, o2++, p1v++, p2v++) {
            __m256d L = _mm256_set1_pd(fast_rand(0.0, 1.0, rng_state));
            __m256d one_minus_L = _mm256_sub_pd(_mm256_set1_pd(1.0), L);
            *o1 = _mm256_add_pd(_mm256_mul_pd(L, *p1v), _mm256_mul_pd(one_minus_L, *p2v));
            *o2 = _mm256_add_pd(_mm256_mul_pd(L, *p2v), _mm256_mul_pd(one_minus_L, *p1v));
        }
    }
    for (; i < dim; i++) {
        double L = fast_rand(0.0, 1.0, rng_state);
        off1[i] = L * p1_pos[i] + (1.0 - L) * p2_pos[i];
        off2[i] = L * p2_pos[i] + (1.0 - L) * p1_pos[i];
    }
    enforce_bounds(off1, bounds, dim);
    enforce_bounds(off2, bounds, dim);
}

// 🦇 Suffocation Phase
void suffocation_phase(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *), int surv_list_size, unsigned int *rng_state) {
    int best_count = 0, best_indices[opt->population_size];
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness == opt->best_solution.fitness) best_indices[best_count++] = i;
    }
    double p_death = best_count > 1 ? (double)(best_count - 1) / opt->population_size : 0.0;

    for (int i = 0; i < best_count; i += 2) {
        if (fast_rand(0.0, 1.0, rng_state) >= p_death) continue;

        if (i == best_count - 1 && best_count % 2 == 1) {
            double *pos = opt->population[best_indices[i]].position;
            replace_with_survival_list(opt, ctx, pos, surv_list_size, rng_state);
            opt->population[best_indices[i]].fitness = objective_function(pos);
            ctx->past_fitness[best_indices[i]] = opt->population[best_indices[i]].fitness;
            update_survival_list(opt, ctx, pos, opt->population[best_indices[i]].fitness, surv_list_size);
        } else {
            int p1 = (int)fast_rand(0.0, opt->population_size, rng_state);
            int p2 = (int)fast_rand(0.0, opt->population_size, rng_state);
            double off1[opt->dim], off2[opt->dim];

            if (fast_rand(0.0, 1.0, rng_state) < 0.5 && 
                opt->population[p1].fitness != opt->population[p2].fitness) {
                crossover(opt, p1, p2, off1, off2, opt->dim, rng_state);
            } else {
                replace_with_survival_list(opt, ctx, off1, surv_list_size, rng_state);
                replace_with_survival_list(opt, ctx, off2, surv_list_size, rng_state);
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
                if (sol->fitness > ctx->worst_fitness) ctx->worst_fitness = sol->fitness;
            }
        }
    }
}

// 🚀 Main Optimization Function
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    unsigned int rng_state = (unsigned int)time(NULL);
    printf("Starting optimization...\n");

    FlyFOContext ctx = { .worst_fitness = -INFINITY, .survival_list = NULL, .survival_list_count = 0 };
    ctx.past_fitness = (double *)malloc(opt->population_size * sizeof(double));

    // Perform actual population initialization
    const int dim = opt->dim, pop_size = opt->population_size;
    double *bounds = opt->bounds;
    Solution *pop = opt->population;
    double best_fitness = INFINITY, worst_fitness = -INFINITY;
    int best_idx = 0;

    for (int i = 0; i < pop_size; i++) {
        double *pos = pop[i].position;
        for (int j = 0; j < dim; j++) {
            pos[j] = bounds[2 * j] + fast_rand(0.0, 1.0, &rng_state) * (bounds[2 * j + 1] - bounds[2 * j]);
        }
        pop[i].fitness = objective_function(pos);
        ctx.past_fitness[i] = pop[i].fitness;
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
    ctx.worst_fitness = worst_fitness;
    printf("Population initialized\n");

    int func_count = pop_size, iteration = 0;
    const int surv_list_size = (int)(pop_size * SURVIVAL_LIST_RATIO);

    static const double deltasO_base[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    double deltasO[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    const double deltasO_min[3] = {DELTASO_0 / 10, DELTASO_1 / 10, DELTASO_2 / 10};

    while (func_count < MAX_EVALS_DEFAULT) {
        iteration++;
        printf("Iteration %d\n", iteration);

        double deltamax = fabs(opt->best_solution.fitness - ctx.worst_fitness);
        double deltas[3] = {deltasO[0] * deltamax, deltasO[1] * deltamax, deltasO[2] * deltamax};

        for (int i = 0; i < pop_size; i++) {
            double alpha, pa;
            fuzzy_self_tuning(opt, &ctx, i, deltamax, &alpha, &pa);
            update_position(opt, &ctx, i, alpha, pa, deltas, objective_function, &rng_state);
            func_count += 2;
        }

        int best_count = 0;
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness == opt->best_solution.fitness) best_count++;
        }
        suffocation_phase(opt, &ctx, objective_function, surv_list_size, &rng_state);
        func_count += best_count;

        for (int k = 0; k < 3; k++) {
            deltasO[k] = deltasO_base[k] - ((deltasO_base[k] - deltasO_min[k]) / MAX_EVALS_DEFAULT) * func_count;
        }

        printf("Function Evaluations = %d, Best Fitness = %f\n", func_count, opt->best_solution.fitness);
    }

    for (int i = 0; i < ctx.survival_list_count; i++) free(ctx.survival_list[i].position);
    free(ctx.survival_list);
    free(ctx.past_fitness);
    printf("Optimization completed\n");
}
