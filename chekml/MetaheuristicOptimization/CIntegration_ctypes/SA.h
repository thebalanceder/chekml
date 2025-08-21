#ifndef SA_H
#define SA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// 🧊 SA Parameters
#define SA_INIT_TEMP       1.0
#define SA_STOP_TEMP       1e-8
#define SA_MAX_TRIES       300
#define SA_MAX_SUCCESS     20
#define SA_MAX_CONS_REJ    1000
#define SA_STOP_VAL       -INFINITY
#define SA_VERBOSITY       1
#define SA_COOLING_RATE    0.8

// 🚀 Main Optimization Function
void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SA_H

