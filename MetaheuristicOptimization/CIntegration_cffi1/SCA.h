#ifndef SCA_H
#define SCA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 SCA Parameters
#define SCA_A 2.0  // Controls linear decrease of r1

// 🚀 SCA Algorithm Functions (Extreme CPU Speed)
void initialize_solutions(Optimizer *opt);
void sca_update_position(Optimizer *opt, double r1_factor);
void SCA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SCA_H
