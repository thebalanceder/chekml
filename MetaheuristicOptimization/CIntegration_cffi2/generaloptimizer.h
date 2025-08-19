#ifndef GENERALOPTIMIZER_H
#define GENERALOPTIMIZER_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // For parallelization (although OpenCL is replacing this)
#include <CL/cl.h> // OpenCL headers

// Define function pointer type for objective functions
typedef double (*ObjectiveFunction)(double*);

// Structure to store a solution (position vector + fitness value)
typedef struct {
    double* position;  // The position vector of the solution
    double fitness;    // The fitness value of the solution
} Solution;

// Optimizer structure
typedef struct {
    int dim;                    // Dimensionality of the problem
    int population_size;        // Number of individuals in the population
    int max_iter;               // Maximum number of iterations
    double* bounds;             // Boundaries for the search space (min, max)
    Solution* population;       // Population array
    Solution best_solution;     // Best solution found
    cl_context context;          // OpenCL context
    cl_command_queue queue;      // OpenCL command queue
    cl_device_id device;         // OpenCL device
    cl_mem population_buffer;    // OpenCL buffer for population
    cl_mem fitness_buffer;       // OpenCL buffer for fitness values
    void (*optimize)(void*, ObjectiveFunction); // Pointer to optimization function
} Optimizer;

void create_kernel(void);
void create_buffers(Optimizer* opt);

// Function declarations
Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method);
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void general_free(Optimizer* opt);
void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness);

// 🛑 **Fix 3:** Ensure boundaries are enforced
void enforce_bound_constraints(Optimizer *opt);

// OpenCL helper functions
void initialize_opencl(Optimizer *opt);
void cleanup_opencl(Optimizer *opt);

// GPU-specific evaluation functions
void evaluate_population_on_gpu(Optimizer* opt, ObjectiveFunction objective_function);
void transfer_population_to_gpu(Optimizer* opt);
void transfer_fitness_from_gpu(Optimizer* opt);

#endif // GENERALOPTIMIZER_H
