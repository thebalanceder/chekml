#ifndef JOA_H
#define JOA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <immintrin.h> // AVX2
#include "generaloptimizer.h"

// Optimization parameters
#define NUM_SUBPOPULATIONS 5
#define POPULATION_SIZE_PER_SUBPOP 10
#define MUTATION_RATE 0.1

// JOA Algorithm Phases
void initialize_subpopulations(Optimizer *opt);
void evaluate_subpopulations(Optimizer *opt, double (*objective_function)(double *));
void update_subpopulations(Optimizer *opt, double *temp_direction);

// Optimization Execution
void JOA_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // JOA_H
