#ifndef GA_H
#define GA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

#define GA_NUM_POPULATIONS 1  // Number of outer population sets
#define GA_VERBOSITY 1        // Set to 0 to disable console logging
#define GA_DEBUG 1            // Set to 1 to enable debug logging

// GA Algorithm Phases
void GA_initialize_population(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                             cl_mem bounds_buffer, int dim, int pop_size, uint seed, cl_event *event);
void GA_crossover(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer,
                  cl_mem best_buffer, cl_mem bounds_buffer, int dim, int pop_size, unsigned int seed, cl_event *event);
void GA_mutate_worst(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                    cl_mem bounds_buffer, cl_mem worst_idx_buffer, int dim, int pop_size, 
                    uint seed, cl_event *event);
void GA_find_best_worst(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, 
                       cl_mem best_idx_buffer, cl_mem worst_idx_buffer, int pop_size, 
                       cl_event *event);

// Optimization Execution
void GA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // GA_H
