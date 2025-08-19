#ifndef TEO_H
#define TEO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>
#include "generaloptimizer.h"

// Optimization parameters
#define TEO_STEP_SIZE 0.5          // Increased for better exploration
#define TEO_INITIAL_TEMPERATURE 100.0
#define TEO_FINAL_TEMPERATURE 0.01
#define TEO_COOLING_RATE 0.95      // Slower cooling for more exploration

// TEO-specific Xorshift RNG state
typedef struct {
    uint64_t state;
} TEO_XorshiftRNG;

// Initialize TEO Xorshift RNG
void teo_xorshift_init(TEO_XorshiftRNG *rng, uint64_t seed);

// Generate random double in [0, 1)
inline double teo_xorshift_double(TEO_XorshiftRNG *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return (x >> 11) * (1.0 / (1ULL << 53));
}

// Generate random double in [min, max)
inline double teo_rand_double(TEO_XorshiftRNG *rng, double min, double max) {
    return min + (max - min) * teo_xorshift_double(rng);
}

// TEO Algorithm Phases
void teo_initialize_solution(Optimizer *restrict opt, double *restrict current_solution, double *restrict current_fitness, double (*objective_function)(double *restrict), TEO_XorshiftRNG *restrict rng);
void teo_perturb_solution(Optimizer *restrict opt, double *restrict current_solution, double *restrict new_solution, TEO_XorshiftRNG *restrict rng);
void teo_accept_solution(Optimizer *restrict opt, double *restrict current_solution, double *restrict new_solution, double new_fitness, double *restrict current_fitness, double temperature, TEO_XorshiftRNG *restrict rng);
void teo_update_best_solution(Optimizer *restrict opt, double *restrict current_solution, double current_fitness);
void teo_cool_temperature(double *restrict temperature);

// Optimization Execution
void TEO_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict));

#ifdef __cplusplus
}
#endif

#endif // TEO_H
