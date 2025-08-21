#ifndef GENERALOPTIMIZER_H
#define GENERALOPTIMIZER_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // For parallelization

// Define function pointer type for objective functions
typedef double (*ObjectiveFunction)(double*);

// Structure to store a solution (position vector + fitness value)
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
    void (*optimize)(void*, ObjectiveFunction);
} Optimizer;

// Function declarations
Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method);
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void general_free(Optimizer* opt);
void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness);

// 🛑 **Fix 3:** Ensure boundaries are enforced
void enforce_bound_constraints(Optimizer *opt);

#endif // GENERALOPTIMIZER_H