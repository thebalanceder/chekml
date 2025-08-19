#ifndef FLYFO_H
#define FLYFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX2
#include "generaloptimizer.h"

// 🔧 Optimization Parameters
#define DELTASO_0 0.2
#define DELTASO_1 0.4
#define DELTASO_2 0.6
#define ALPHA_PARAM_0 1.0
#define ALPHA_PARAM_1 1.5
#define ALPHA_PARAM_2 1.9
#define PA_PARAM_0 0.5
#define PA_PARAM_1 0.85
#define PA_PARAM_2 0.99
#define SURVIVAL_LIST_RATIO 0.25
#define DEATH_PROB_FACTOR 1.0

// ⚙️ Algorithm Constants
#define MAX_EVALS_DEFAULT 100
#define POPULATION_SIZE_FACTOR 10
#define SQRT_DIM_FACTOR 2

// 🦇 Aligned FFO Context
typedef struct {
    double worst_fitness;
    Solution *survival_list;
    int survival_list_count;
    double *past_fitness;
} __attribute__((aligned(64))) FlyFOContext;

// 🌟 Inline Utility Functions
inline double fast_rand(double min, double max, unsigned int *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return min + (max - min) * ((double)*state / 0xFFFFFFFFU);
}

inline void enforce_bounds(double *pos, const double *bounds, int dim) {
    int i = 0;
    if (dim >= 4) {
        __m256d *p = (__m256d *)pos;
        const __m256d *b_lower = (const __m256d *)bounds;
        const __m256d *b_upper = (const __m256d *)(bounds + 2);
        for (; i <= dim - 4; i += 4, p++, b_lower++, b_upper += 2) {
            __m256d v = _mm256_loadu_pd((double *)p);
            __m256d lower = _mm256_loadu_pd((double *)b_lower);
            __m256d upper = _mm256_loadu_pd((double *)b_upper);
            v = _mm256_max_pd(v, lower);
            v = _mm256_min_pd(v, upper);
            _mm256_storeu_pd((double *)p, v);
        }
    }
    for (; i < dim; i++) {
        if (pos[i] < bounds[2 * i]) pos[i] = bounds[2 * i];
        if (pos[i] > bounds[2 * i + 1]) pos[i] = bounds[2 * i + 1];
    }
}

inline void fuzzy_self_tuning(const Optimizer *opt, FlyFOContext *ctx, int i, double deltamax, double *alpha, double *pa) {
    double delta = fabs(opt->best_solution.fitness - opt->population[i].fitness);
    double fi = deltamax > 0.0 ? (opt->population[i].fitness - ctx->past_fitness[i]) / deltamax : 0.0;

    static const double deltas_base[3] = {DELTASO_0, DELTASO_1, DELTASO_2};
    static const double alpha_params[3] = {ALPHA_PARAM_0, ALPHA_PARAM_1, ALPHA_PARAM_2};
    static const double pa_params[3] = {PA_PARAM_0, PA_PARAM_1, PA_PARAM_2};
    double deltas[3] = {deltas_base[0] * deltamax, deltas_base[1] * deltamax, deltas_base[2] * deltamax};

    double delta_membership[3] = {0.0};
    double fi_membership[3] = {0.0, 1.0 - fabs(fi), 0.0};

    if (delta < deltas[1]) {
        if (delta < deltas[0]) delta_membership[0] = 1.0;
        else {
            delta_membership[0] = (deltas[1] - delta) / (deltas[1] - deltas[0]);
            delta_membership[1] = (delta - deltas[0]) / (deltas[1] - deltas[0]);
        }
    } else if (delta <= deltamax) {
        if (delta <= deltas[2]) {
            delta_membership[1] = (deltas[2] - delta) / (deltas[2] - deltas[1]);
            delta_membership[2] = (delta - deltas[1]) / (deltas[2] - deltas[1]);
        } else delta_membership[2] = 1.0;
    }

    if (fi >= -1.0 && fi <= 1.0) {
        if (fi < 0.0) fi_membership[0] = -fi;
        else if (fi > 0.0) fi_membership[2] = fi;
    }

    double ruleno_alpha[3] = {
        fi_membership[0],
        fmax(fi_membership[1], fmax(delta_membership[0], delta_membership[1])),
        fmax(fi_membership[2], delta_membership[2])
    };
    double alpha_sum = ruleno_alpha[0] + ruleno_alpha[1] + ruleno_alpha[2];
    *alpha = alpha_sum > 0.0 ? (ruleno_alpha[0] * alpha_params[0] + 
                                ruleno_alpha[1] * alpha_params[1] + 
                                ruleno_alpha[2] * alpha_params[2]) / alpha_sum : 1.0;

    double ruleno_pa[3] = {
        fmax(fi_membership[2], delta_membership[2]),
        fmax(fi_membership[1], delta_membership[0]),
        fmax(fi_membership[0], delta_membership[1])
    };
    double pa_sum = ruleno_pa[0] + ruleno_pa[1] + ruleno_pa[2];
    *pa = pa_sum > 0.0 ? (ruleno_pa[0] * pa_params[0] + 
                          ruleno_pa[1] * pa_params[1] + 
                          ruleno_pa[2] * pa_params[2]) / pa_sum : 0.85;
}

// 🦇 FFO Algorithm Phases
void initialize_population_flyfo(Optimizer *opt);
void update_position(Optimizer *opt, FlyFOContext *ctx, int i, double alpha, double pa, const double *deltas, double (*objective_function)(double *), unsigned int *rng_state);
void update_survival_list(Optimizer *opt, FlyFOContext *ctx, const double *pos, double fitness, int surv_list_size);
void replace_with_survival_list(Optimizer *opt, FlyFOContext *ctx, double *pos, int surv_list_size, unsigned int *rng_state);
void crossover(const Optimizer *opt, int p1, int p2, double *off1, double *off2, int dim, unsigned int *rng_state);
void suffocation_phase(Optimizer *opt, FlyFOContext *ctx, double (*objective_function)(double *), int surv_list_size, unsigned int *rng_state);

// 🚀 Optimization Execution
void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FLYFO_H
