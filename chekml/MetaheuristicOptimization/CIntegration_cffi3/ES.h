#ifndef EAGLE_STRATEGY_H
#define EAGLE_STRATEGY_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Cognitive and Social Coefficients
#define C1 2.0f
#define C2 2.0f

// Inertia Weight
#define W_MAX 0.9f
#define W_MIN 0.4f

// Lévy flight parameters
#define LEVY_BETA 1.5f
#define LEVY_STEP_SCALE 0.1f
#define LEVY_PROBABILITY 0.2f

// ES Algorithm Phases
void initialize_es(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem velocity_buffer, 
                   cl_mem local_best_buffer, cl_mem local_best_cost_buffer, cl_mem global_best_buffer, 
                   cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event);
void update_velocity_and_position(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                                 cl_mem velocity_buffer, cl_mem local_best_buffer, cl_mem global_best_buffer, 
                                 cl_mem bounds_buffer, int dim, int population_size, int iter, int max_iter, 
                                 uint seed, cl_event *event);
void levy_flight(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem local_best_buffer, 
                 cl_mem global_best_buffer, cl_mem bounds_buffer, int dim, int population_size, uint seed, 
                 cl_event *event);
void find_best(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, cl_mem best_idx_buffer, 
               int population_size, cl_event *event);

// Main function
void ES_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // EAGLE_STRATEGY_H
