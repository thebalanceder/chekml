#ifndef MFA_H
#define MFA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MFA_POP_SIZE 100
#define MFA_DIM_MAX 100
#define MFA_NUM_OFFSPRING 20
#define MFA_NUM_MUTANTS 1
#define MFA_INERTIA_WEIGHT 0.8
#define MFA_INERTIA_DAMP 1.0
#define MFA_PERSONAL_COEFF 1.0
#define MFA_GLOBAL_COEFF1 1.5
#define MFA_GLOBAL_COEFF2 1.5
#define MFA_DISTANCE_COEFF 2.0
#define MFA_NUPTIAL_DANCE 5.0
#define MFA_RANDOM_FLIGHT 1.0
#define MFA_DANCE_DAMP 0.8
#define MFA_FLIGHT_DAMP 0.99
#define MFA_MUTATION_RATE 0.01

// Fast random number generator
void xorshift_seed_mfa(unsigned int seed);
static inline unsigned int xorshift32_mfa(void);
static inline double rand_double_mfa(double min, double max);

// Mayfly Algorithm Phases
void mfa_initialize_populations(Optimizer *opt, double (*objective_function)(double *));
void mfa_update_males(Optimizer *opt, double (*objective_function)(double *));
void mfa_update_females(Optimizer *opt, double (*objective_function)(double *));
void mfa_mating_phase(Optimizer *opt, double (*objective_function)(double *));
void mfa_mutation_phase(Optimizer *opt, double (*objective_function)(double *));
void mfa_sort_and_select(Optimizer *opt, Solution *population, int pop_size);

// Main Optimization Function
void MfA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // MFA_H
