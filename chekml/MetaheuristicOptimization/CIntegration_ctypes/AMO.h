#ifndef AMO_H
#define AMO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization Parameters
#define NEIGHBORHOOD_SIZE_AMO 5
#define POPULATION_SCALE_FACTOR 1.0
#define MIGRATION_PROBABILITY_FACTOR 0.5

// ⚙️ Algorithm Constants
#define NEIGHBORHOOD_RADIUS 2
#define FITNESS_SCALING 1.0

double normal_rand(double mean, double stddev);

// 🌍 AMO Algorithm Phases
void initialize_population_amo(Optimizer *opt);
void neighborhood_learning_phase(Optimizer *opt);
void global_migration_phase(Optimizer *opt);

// 🚀 Optimization Execution
void AMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // AMO_H
