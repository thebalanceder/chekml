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

// TLBO Algorithm Phases
void teacher_phase(Optimizer *opt, ObjectiveFunction objective_function);
void learner_phase(Optimizer *opt, ObjectiveFunction objective_function);

// Main Optimization Function
void TLBO_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // TLBO_H
