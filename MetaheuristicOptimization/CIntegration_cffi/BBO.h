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
#define MUTATION_PROB 0.1
#define MUTATION_SCALE 0.02

// BBO-specific data structure
typedef struct {
    double *mu;              // Emigration rates
    double *lambda_;         // Immigration rates
    double *ep_buffer;       // Reusable buffer for emigration probabilities
    double mutation_sigma;   // Mutation standard deviation
    int store_history;       // Flag to enable/disable history storage
    struct {
        int iteration;
        double *solution;
        double value;
    } *history;              // History of best solutions
} BBOData;

// BBO Algorithm Phases
void bbo_initialize_habitats(Optimizer *opt, BBOData *data);
void bbo_migration_phase(Optimizer *opt, BBOData *data);
void bbo_selection_phase(Optimizer *opt, BBOData *data);

// Main Optimization Function
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BBO_H
