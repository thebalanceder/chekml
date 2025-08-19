#ifndef FA_H
#define FA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>    // ✅ Ensure rand(), srand() work properly
#include <stdio.h>     // ✅ For debugging/logging
#include <stdint.h>    // ✅ For uint64_t
#include <immintrin.h> // ✅ For AVX2 intrinsics
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define NUM_PARTICLES 50
#define MAX_ITER 100
#define ALPHA 0.1
#define BETA 1.0
#define DELTA_T 1.0

// 🎲 Xorshift RNG state
typedef struct {
    uint64_t state;
} XorshiftState_Fa;

// 🎲 Random number generation
void xorshift_seed(XorshiftState_Fa *state, uint64_t seed);
double xorshift_double_fa(XorshiftState_Fa *state, double min, double max);
double xorshift_gaussian(XorshiftState_Fa *state);

// 🌠 FWA Algorithm Phases
void initialize_particles(Optimizer *opt, XorshiftState_Fa *rng);
void evaluate_fitness(Optimizer *opt, double (*objective_function)(double *));
void generate_sparks(Optimizer *opt, double *best_particle, double *sparks, XorshiftState_Fa *rng);
void update_particles(Optimizer *opt, double (*objective_function)(double *), double *all_positions, double *all_fitness, void *sort_array, XorshiftState_Fa *rng);

// 🚀 Optimization Execution
void FA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FA_H
