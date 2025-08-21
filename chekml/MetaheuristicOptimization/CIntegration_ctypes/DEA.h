#ifndef DEA_H
#define DEA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// DEA Optimization Parameters
#define DEA_POPULATION_SIZE 50
#define DEA_MAX_LOOPS 100
#define DEA_CONVERGENCE_POWER 1.0
#define DEA_EFFECTIVE_RADIUS_FACTOR 0.25
#define DEA_PROBABILITY_THRESHOLD 0.1
#define DEA_ALTERNATIVES_PER_DIM 100

// Structure to hold alternatives and accumulative fitness per dimension
typedef struct {
    double *values; // Discretized alternative values
    double *accum_fitness; // Accumulative fitness for alternatives
    double *probabilities; // Probabilities for alternatives
    int size; // Number of alternatives
} DimensionData;

// Global DEA data
typedef struct {
    DimensionData *dim_data; // Per-dimension alternatives and fitness
    double effective_radius; // Effective radius for fitness distribution
} DEAData;

// DEA Algorithm Phases
void initialize_locations(Optimizer *opt, DEAData *dea_data);
void calculate_accumulative_fitness(Optimizer *opt, double *fitness, DEAData *dea_data);
double get_convergence_probability(int loop, int max_loops);
void update_probabilities(Optimizer *opt, int loop, double *fitness, DEAData *dea_data);
void select_new_locations(Optimizer *opt, DEAData *dea_data);
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DEA_H
