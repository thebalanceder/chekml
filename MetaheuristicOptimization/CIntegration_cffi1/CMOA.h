#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 CMOA Parameters
#define CMOA_MUTATION_RATE 0.3
#define CMOA_CROSSOVER_RATE 0.5
#define CMOA_MUTATION_SCALE 0.1  // Scale for non-genetic mutation range

// 🦠 CMOA Algorithm Phases
void genetic_recombination(Optimizer *opt, double *randoms);
void cross_activation(Optimizer *opt, double *randoms);
void incremental_reactivation(Optimizer *opt, int t, double *randoms);
void non_genetic_mutation(Optimizer *opt, double *randoms);
void genotypic_mixing(Optimizer *opt, int *random_indices);

// 🚀 Optimization Execution
void CMOA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

