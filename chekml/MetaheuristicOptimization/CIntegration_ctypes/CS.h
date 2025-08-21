#ifndef CS_H
#define CS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

// 🔧 Parameters
#define MUTATION_PROBABILITY 0.1
#define CLONE_FACTOR 0.1
#define REPLACEMENT_RATE 0.1

// 🚀 Main Optimization Function
void CS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // CS_H

