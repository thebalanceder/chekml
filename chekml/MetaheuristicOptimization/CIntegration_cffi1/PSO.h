#ifndef PSO_H
#define PSO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h> // SSE/SSE2
#include <smmintrin.h> // SSE4.1/SSE4.2
#include <omp.h>
#include "generaloptimizer.h"

// 🔧 PSO Parameters (Tuned for Extreme Speed)
#define INERTIA_WEIGHT 1.0
#define INERTIA_DAMPING 0.99
#define PERSONAL_LEARNING 1.5
#define GLOBAL_LEARNING 2.0
#define VELOCITY_SCALE 0.1

// 🌠 PSO Algorithm Phases
void PSO_initialize_swarm(Optimizer *opt, double *velocities, double *pbest_positions, double *pbest_fitnesses);
void PSO_update_velocity_position(Optimizer *opt, double *velocities, double *pbest_positions);
void PSO_evaluate_particles(Optimizer *opt, double (*objective_function)(double *), 
                           double *pbest_positions, double *pbest_fitnesses);

// 🚀 Optimization Execution
void PSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // PSO_H
