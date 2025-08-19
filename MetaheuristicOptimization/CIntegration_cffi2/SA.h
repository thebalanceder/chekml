#ifndef SA_H
#define SA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// 🧊 SA Parameters
#define INITIAL_TEMP      1.0
#define STOP_TEMP         1e-8
#define MAX_TRIES         300
#define MAX_SUCCESS       20
#define MAX_CONSEC_REJ    1000
#define BOLTZMANN_CONST   1.0
#define MIN_DELTA         1e-6
#define COOLING_FACTOR    0.8

// 🚀 Main Optimization Function
void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SA_H

