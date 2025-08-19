// KA.h
#ifndef KA_H
#define KA_H

#include "generaloptimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

void KA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // KA_H

