#ifndef MFA_H
#define MFA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
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
#define MFA_NUM_OFFSPRING 20
#define MFA_NUM_MUTANTS 1
#define MFA_MUTATION_RATE 0.01

// Random number generation
double rand_double_mfa(double min, double max);

// Mayfly Algorithm Phases
void mfa_initialize_populations(Optimizer *opt, double (*objective_function)(double *));
void mfa_update_males(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                     double inertia_weight, double personal_coeff, double global_coeff1, double distance_coeff, 
                     double nuptial_dance);
void mfa_update_females(Optimizer *opt, double (*objective_function)(double *), double *vel_max, double *vel_min, 
                       double inertia_weight, double global_coeff2, double distance_coeff, double random_flight);
void mfa_mating_phase(Optimizer *opt, double (*objective_function)(double *), double *bounds);
void mfa_mutation_phase(Optimizer *opt, double (*objective_function)(double *), double *bounds, 
                       double mutation_rate);
void mfa_sort_and_select(Optimizer *opt, Solution *population, int pop_size);

// Main Optimization Function
void MfA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // MFA_H
