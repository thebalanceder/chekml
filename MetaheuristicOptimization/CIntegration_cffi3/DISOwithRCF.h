#ifndef DISO_WITH_RCF_H
#define DISO_WITH_RCF_H

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
#define DIVERSION_FACTOR 0.3f
#define FLOW_ADJUSTMENT 0.2f
#define WATER_DENSITY 1.35f
#define FLUID_DISTRIBUTION 0.46f
#define CENTRIFUGAL_RESISTANCE 1.2f
#define BOTTLENECK_RATIO 0.68f
#define ELIMINATION_RATIO 0.23f

// Hydrodynamic Constants
#define HRO 1.2f
#define HRI 7.2f
#define HGO 1.3f
#define HGI 0.82f
#define CFR_FACTOR 9.435f

// DISO Algorithm Phases
void diversion_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem best_solution_buffer, 
                     cl_mem bounds_buffer, int dim, int population_size, uint seed, cl_event *event);
void spiral_motion_update(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem fitness_buffer, 
                         cl_mem best_solution_buffer, cl_mem bounds_buffer, int dim, int population_size, int iter, 
                         int max_iter, uint seed, cl_event *event);
void local_development_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, 
                            cl_mem best_solution_buffer, cl_mem bounds_buffer, int dim, int population_size, 
                            uint seed, cl_event *event);
void elimination_phase(cl_kernel kernel, cl_command_queue queue, cl_mem population_buffer, cl_mem bounds_buffer, 
                       int dim, int population_size, uint seed, cl_event *event);

// Optimization Execution
void DISO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DISO_WITH_RCF_H
