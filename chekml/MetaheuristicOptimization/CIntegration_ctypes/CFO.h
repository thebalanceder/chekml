#ifndef CFO_H
#define CFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define ALPHA 0.1

// 🚀 CFO Algorithm Phases
void initialize_population_cfo(Optimizer *opt);
void central_force_update(Optimizer *opt);
void update_best_solution_cfo(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void CFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CFO_H