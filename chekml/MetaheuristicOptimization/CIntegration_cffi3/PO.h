#ifndef POLITICAL_OPTIMIZER_H
#define POLITICAL_OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

void PO_optimize(void *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // POLITICAL_OPTIMIZER_H
