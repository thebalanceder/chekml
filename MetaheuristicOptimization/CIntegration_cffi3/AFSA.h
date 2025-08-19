#ifndef AFSA_H
#define AFSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "generaloptimizer.h"

// AFSA Parameters
#define VISUAL 0.3f
#define STEP_SIZE_AFSA 0.1f
#define TRY_NUMBER 5
#define DELTA 0.618f

// AFSA API
void AFSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // AFSA_H
