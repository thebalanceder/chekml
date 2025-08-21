#ifndef HBO_H
#define HBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include "generaloptimizer.h"

// 🚀 Main Optimization Execution
void HBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HBO_H

