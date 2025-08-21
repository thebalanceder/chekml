/* AFSA.h - Header file for Artificial Fish Swarm Algorithm */
#ifndef AFSA_H
#define AFSA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include "generaloptimizer.h"

// AFSA Parameters
#define VISUAL 0.3
#define STEP_SIZE 0.1
#define TRY_NUMBER 5
#define DELTA 0.618

// Core AFSA Behaviors
void prey_behavior(Optimizer *opt, int index, double (*objective_function)(double *));
void swarm_behavior(Optimizer *opt, int index, double (*objective_function)(double *));
void follow_behavior(Optimizer *opt, int index, double (*objective_function)(double *));

double rand_double(double min, double max);

// Main Optimization
void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // AFSA_H
