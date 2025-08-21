#ifndef BBO_H
#define BBO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define KEEP_RATE 0.2
#define BBO_ALPHA 0.9
#define BBO_MUTATION_PROB 0.1
#define MUTATION_SCALE 0.02

// BBO-specific data structure
typedef struct {
    double *mu;              // Emigration rates
    double *lambda_;         // Immigration rates
    double *ep_buffer;       // Reusable buffer for emigration probabilities
    double *population_data; // Contiguous storage for population positions
    double *random_buffer;   // Precomputed Gaussian random numbers
    double mutation_sigma;   // Mutation standard deviation
    int random_buffer_size;  // Size of random buffer
    int random_buffer_idx;   // Current index in random buffer
} BBOData;

// BBO Algorithm Phases
void bbo_initialize_habitats(Optimizer *opt, BBOData *data);
void bbo_migration_phase(Optimizer *opt, BBOData *data, double (*objective_function)(double *));
void bbo_selection_phase(Optimizer *opt, BBOData *data);

// Main Optimization Function
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BBO_H
