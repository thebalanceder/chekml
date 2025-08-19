#ifndef SA_H
#define SA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// 🧊 SA Parameters
#define INITIAL_TEMP      10.0       // Increased for better exploration
#define STOP_TEMP         1e-8
#define MAX_TRIES         1000
#define MAX_SUCCESS       20
#define MAX_CONSEC_REJ    1000
#define BOLTZMANN_CONST   1.0
#define MIN_DELTA         1e-6
#define COOLING_FACTOR    0.9        // Slower cooling
#define CANDIDATE_COUNT   1024
#define MAX_ITER_SA          100

// 🚀 Main Optimization Function
void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SA_H
