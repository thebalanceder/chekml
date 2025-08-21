#ifndef SWO_H
#define SWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define TRADE_OFF 0.3
#define CROSSOVER_PROB 0.2
#define MIN_POPULATION 20
#define LEVY_BETA 1.5
#define LEVY_SCALE 0.05

// SWO Algorithm Phases
void SWO_hunting_phase(Optimizer *opt, double (*objective_function)(double *));
void SWO_mating_phase(Optimizer *opt, double (*objective_function)(double *));
void SWO_population_reduction(Optimizer *opt, int iter);

// Optimization Execution
void SWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SWO_H
