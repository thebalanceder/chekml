#ifndef VNS_H
#define VNS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "generaloptimizer.h"

#define MUTATION_RATE 0.1

typedef double (*ObjectiveFunction)(double*);

// Optimization execution
void VNS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // VNS_H
