#ifndef GENERALOPTIMIZER_H
#define GENERALOPTIMIZER_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <CL/cl.h>

// Define function pointer type for objective functions
typedef double (*ObjectiveFunction)(double*);

// Structure to store a solution
typedef struct {
    double* position;
    double fitness;
} Solution;

// Optimizer structure
typedef struct {
    int dim;
    int population_size;
    int max_iter;
    double* bounds;
    Solution* population;
    Solution best_solution;
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_mem population_buffer;
    cl_mem fitness_buffer;
    void (*optimize)(void*, ObjectiveFunction);
} Optimizer;

// Function declarations
Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method);
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void general_free(Optimizer* opt);
void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness);
void enforce_bound_constraints(Optimizer* opt);
void initialize_opencl(Optimizer* opt);
void create_buffers(Optimizer* opt);
void cleanup_opencl(Optimizer* opt);
void generate_points_on_gpu(Optimizer* opt, int points_per_dim, int total_points);

#endif // GENERALOPTIMIZER_H
