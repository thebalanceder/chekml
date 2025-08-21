#ifndef FA_H
#define FA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure rand(), srand() work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define NUM_PARTICLES 50
#define MAX_ITER 100
#define ALPHA 0.1
#define BETA 1.0
#define DELTA_T 1.0

// 🎲 Random number generation
double rand_gaussian_fa();

// 🌠 FWA Algorithm Phases
void initialize_particles(Optimizer *opt);
void evaluate_fitness(Optimizer *opt, double (*objective_function)(double *));
void generate_sparks(Optimizer *opt, double *best_particle, double *sparks);
void update_particles(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void FA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FA_H
