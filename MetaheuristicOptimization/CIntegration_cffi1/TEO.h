#ifndef TEO_H
#define TEO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "generaloptimizer.h"

// Optimization parameters
#define TEO_STEP_SIZE 0.1
#define TEO_INITIAL_TEMPERATURE 100.0
#define TEO_FINAL_TEMPERATURE 0.01
#define TEO_COOLING_RATE 0.99

// TEO-specific Xorshift random number generator
typedef struct {
    uint64_t state;
} TEO_XorshiftRNG;

inline uint64_t teo_xorshift_next(TEO_XorshiftRNG *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

inline double teo_fast_rand_double(TEO_XorshiftRNG *rng, double min, double max) {
    return min + (max - min) * ((double)teo_xorshift_next(rng) / UINT64_MAX);
}

// Fast approximate exp() for acceptance criterion
inline double teo_fast_exp(double x) {
    if (x < -10.0) return 0.0;
    double x2 = x * x;
    return 1.0 + x + (x2 / 2.0) + (x2 * x / 6.0);
}

// TEO Algorithm Phases
inline void initialize_solution(Optimizer *opt, double *solution, double *fitness, double (*objective_function)(double *), TEO_XorshiftRNG *rng) {
    for (int j = 0; j < opt->dim; j++) {
        double min = opt->bounds[2 * j];
        double max = opt->bounds[2 * j + 1];
        solution[j] = min + (max - min) * teo_fast_rand_double(rng, 0.0, 1.0);
        solution[j] = solution[j] < min ? min : (solution[j] > max ? max : solution[j]);
    }
    *fitness = objective_function(solution);
    
    opt->best_solution.fitness = *fitness;
    memcpy(opt->best_solution.position, solution, opt->dim * sizeof(double));
}

inline void perturb_solution(Optimizer *opt, double *new_solution, double step_size, TEO_XorshiftRNG *rng) {
    #pragma omp simd
    for (int j = 0; j < opt->dim; j++) {
        double perturbation = step_size * teo_fast_rand_double(rng, -1.0, 1.0);
        new_solution[j] = opt->best_solution.position[j] + perturbation;
        double min = opt->bounds[2 * j];
        double max = opt->bounds[2 * j + 1];
        new_solution[j] = new_solution[j] < min ? min : (new_solution[j] > max ? max : new_solution[j]);
    }
}

inline void accept_solution(Optimizer *opt, double *new_solution, double new_fitness, double temperature, TEO_XorshiftRNG *rng) {
    double delta_fitness = new_fitness - opt->best_solution.fitness;
    if (delta_fitness <= 0 || teo_fast_rand_double(rng, 0.0, 1.0) < teo_fast_exp(-delta_fitness / temperature)) {
        memcpy(opt->best_solution.position, new_solution, opt->dim * sizeof(double));
        opt->best_solution.fitness = new_fitness;
    }
}

// Optimization Execution
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TEO_H
