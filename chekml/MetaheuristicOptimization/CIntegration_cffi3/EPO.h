#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include <CL/cl.h>
#include "generaloptimizer.h"
#define INITIAL_F 2.0f
#define INITIAL_L 1.5f
#define INITIAL_M 0.5f
#define ADAPTATION_INTERVAL 10
#define STRATEGY_COUNT 3
#define STRATEGY_LINEAR 0
#define STRATEGY_EXPONENTIAL 1
#define STRATEGY_CHAOTIC 2
void EPO_optimize(Optimizer *opt, ObjectiveFunction objective_function);
#ifdef __cplusplus
}
#endif
