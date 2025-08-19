#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CRO_INITIAL_KE 1000.0
#define CRO_MOLE_COLL 0.5
#define CRO_BUFFER_INITIAL 0.0
#define CRO_ALPHA 10.0
#define CRO_BETA 0.2
#define CRO_SPLIT_RATIO 0.5
#define CRO_ELIMINATION_RATIO 0.2

// Molecule structure
typedef struct {
    double *position;    // Solution vector
    double pe;           // Potential energy (fitness)
    double ke;           // Kinetic energy
} Molecule;

// CRO-specific optimizer structure
typedef struct {
    Optimizer *opt;                // Base optimizer
    Molecule *molecules;           // Population of molecules
    int population_size;           // Current[Current population size
    double buffer;                 // Buffer energy
    double (*objective_function)(double *); // Objective function
} CROOptimizer;

// Function prototypes
void initialize_molecules(CROOptimizer *cro);
void evaluate_molecules(CROOptimizer *cro);
void on_wall_collision(CROOptimizer *cro, int index);
void decomposition(CROOptimizer *cro, int index);
void inter_molecular_collision(CROOptimizer *cro, int index1, int index2);
void synthesis(CROOptimizer *cro, int index1, int index2);
void elimination_phase_chero(CROOptimizer *cro);
void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif
