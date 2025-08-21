#ifndef GWCA_H
#define GWCA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// GWCA-specific constants
#define G 9.8f      // Gravitational constant (m/s^2)
#define M 3.0f      // Method constant
#define E 0.1f      // Method constant
#define P 9.0f      // Method constant
#define Q 6.0f      // Method constant
#define CMAX 20.0f  // Max value for constant C
#define CMIN 10.0f  // Min value for constant C

// GWCA-specific parameters
#define GWCA_POPULATION_SIZE 256
#define GWCA_LOCAL_WORK_SIZE 64

// GWCA optimization function
void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // GWCA_H
