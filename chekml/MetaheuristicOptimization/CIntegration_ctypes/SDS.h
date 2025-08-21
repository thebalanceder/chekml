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

// 🔧 Optimization parameters (from sds_algorithm.pyx)
#define SDS_MUTATION_RATE 0.08   // Probability of mutation in diffusion
#define SDS_MUTATION_SCALE 4.0   // Controls mutation offset standard deviation
#define SDS_CLUSTER_THRESHOLD 0.33 // Fraction of agents for convergence
#define SDS_CONTEXT_SENSITIVE 0  // Context-sensitive diffusion (0 = False)

// ⚙️ SDS Constants
#define SDS_CONVERGENCE_TOLERANCE 1e-3 // Distance threshold for clustering
#define SDS_MAX_COMPONENTS 10    // Maximum number of component functions

// 🌐 Function Declarations
double rand_double_kca(double min, double max);
void initialize_agents(Optimizer *opt);
int evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx);
void test_phase(Optimizer *opt, int *activities, double (*objective_function)(double *));
void diffusion_phase(Optimizer *opt, int *activities);
double evaluate_full_objective(double (*objective_function)(double *), double *hypothesis);
int check_convergence(Optimizer *opt);
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SDS_H
