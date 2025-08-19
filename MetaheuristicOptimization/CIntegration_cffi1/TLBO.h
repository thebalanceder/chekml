#ifndef TLBO_H
#define TLBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Main Optimization Function
void TLBO_optimize(Optimizer *restrict opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // TLBO_H
