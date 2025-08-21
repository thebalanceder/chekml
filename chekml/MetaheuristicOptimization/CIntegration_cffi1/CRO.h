#ifndef CRO_H
#define CRO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define POPULATION_SIZE 50
#define MAX_ITER 100
#define NUM_REEFS 10
#define ALPHA 0.1

// ⚙️ Toggle OpenMP
#ifndef _OPENMP
#define NO_OPENMP
#endif

// ⚙️ Function prototypes
void initialize_reefs(Optimizer *opt);
void evaluate_reefs(Optimizer *opt, double (*objective_function)(double *), int *modified, int modified_count);
void migration_phase_cro(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef);
void local_search_phase(Optimizer *opt, int *modified, int *modified_count, int solutions_per_reef);
void CRO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CRO_H
