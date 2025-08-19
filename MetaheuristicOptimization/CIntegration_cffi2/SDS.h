#pragma once

#ifndef SDS_H
#define SDS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters (tuned for faster convergence)
#define SDS_MUTATION_RATE 0.1    // Increased for more diversity
#define SDS_MUTATION_SCALE 3.0   // Reduced for finer mutations
#define SDS_CLUSTER_THRESHOLD 0.4 // Higher threshold for stricter convergence
#define SDS_CONTEXT_SENSITIVE 0  // Context-sensitive diffusion (0 = False)

// ⚙️ SDS Constants
#define SDS_CONVERGENCE_TOLERANCE 1e-3 // Distance threshold for clustering
#define SDS_MAX_COMPONENTS 10    // Maximum number of component functions
#define SDS_INV_MUTATION_SCALE (1.0 / SDS_MUTATION_SCALE) // Precomputed reciprocal

// 🌐 Function Declarations
double rand_double_sds(double min, double max);
double rand_normal_sds(void);
void initialize_agents(Optimizer *opt);
int evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx);
void test_phase(Optimizer *opt, int *activities, double (*objective_function)(double *));
void diffusion_phase(Optimizer *opt, int *activities, int iter, int max_iter);
double evaluate_full_objective(double (*objective_function)(double *), double *hypothesis);
int check_convergence(Optimizer *opt, double prev_best_fitness);
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SDS_H
