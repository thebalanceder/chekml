#ifndef SO_H
#define SO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define SPIRAL_STEP 0.1f  // Controls the step size in spiral movement

// SO Algorithm Phases
void SO_initialize_population(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event);
void SO_spiral_movement_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event);
void SO_find_best(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, 
                  cl_mem best_idx_buffer, int population_size, cl_event *event);

// Optimization Execution
void SO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SO_H
