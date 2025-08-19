#ifndef TABU_SEARCH_H
#define TABU_SEARCH_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

void TS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TABU_SEARCH_H

