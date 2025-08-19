#ifndef CROSA_H
#define CROSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define AWARENESS_PROBABILITY 0.1
#define FLIGHT_LENGTH 2.0

// 🌍 Utility function for random number generation
#define RAND_DOUBLE(min, max) ((min) + ((max) - (min)) * ((double)rand() / RAND_MAX))

// 🐦 CSA Algorithm Phases
void initialize_population_crosa(Optimizer *opt, Solution *memory, double (*objective_function)(double *));
void update_positions(Optimizer *opt, Solution *memory);
void update_memory(Optimizer *opt, Solution *memory, double (*objective_function)(double *));

// 🚀 Optimization Execution
void CroSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CROSA_H
