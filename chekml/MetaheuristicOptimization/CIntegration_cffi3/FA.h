#ifndef FA_H
#define FA_H

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
#define NUM_PARTICLES 50
#define MAX_ITER 100
#define ALPHA 0.1f
#define BETA 1.0f
#define DELTA_T 1.0f

// FWA Algorithm Phases
void initialize_particles(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                         cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event);
void generate_sparks(cl_kernel kernel, cl_command_queue queue, cl_mem sparks_buffer, 
                     cl_mem best_particle_buffer, cl_mem bounds_buffer, int dim, int num_sparks, 
                     uint seed, cl_event *event);
void combine_and_constrain(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                          cl_mem sparks_buffer, cl_mem all_positions_buffer, cl_mem bounds_buffer, 
                          int dim, int population_size, int num_sparks, cl_event *event);
void find_best_fa(cl_kernel kernel, cl_command_queue queue, cl_mem fitness_buffer, cl_mem best_idx_buffer, 
               int total_particles, cl_event *event);

// Optimization Execution
void FA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FA_H
