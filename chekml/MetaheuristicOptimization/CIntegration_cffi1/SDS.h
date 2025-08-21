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

// 🔧 Optimization parameters (tuned for speed and convergence)
#define SDS_MUTATION_RATE 0.12   // Higher for diversity
#define SDS_MUTATION_SCALE 2.5   // Tighter mutations
#define SDS_CLUSTER_THRESHOLD 0.45 // Strict convergence
#define SDS_CONTEXT_SENSITIVE 0  // Disable for speed
#define SDS_CONVERGENCE_TOL_SQ (1e-3 * 1e-3) // Squared tolerance
#define SDS_MAX_COMPONENTS 10    // Component limit
#define SDS_INV_MUTATION_SCALE (1.0 / SDS_MUTATION_SCALE)
#define SDS_STAGNATION_TOL 1e-7  // Stagnation threshold

// 🌐 Function Declarations
double sds_rand_double(void);
double sds_rand_normal(void);
void sds_initialize_agents(Optimizer *opt);
inline int sds_evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx);
void sds_test_phase(Optimizer *opt, int *activities, double (*objective_function)(double *));
void sds_diffusion_phase(Optimizer *opt, int *activities, int iter);
inline double sds_evaluate_full_objective(double (*objective_function)(double *), double *hypothesis);
int sds_check_convergence(Optimizer *opt, double prev_best_fitness);
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *));

// Inline Definitions
inline int sds_evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx) {
    double value = objective_function(hypothesis);
    double t = fabs(value) / fmax(1.0, fabs(value));
    return (sds_rand_double() < t) ? 0 : 1;
}

inline double sds_evaluate_full_objective(double (*objective_function)(double *), double *hypothesis) {
    return objective_function(hypothesis);
}

#ifdef __cplusplus
}
#endif

#endif // SDS_H
