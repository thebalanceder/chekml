#ifndef EVO_H
#define EVO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define STEP_SIZE 0.1
#define MOMENTUM 0.9
#define LEARNING_RATE 0.2

// Structure to hold EVO-specific particle data (velocity and gradient)
typedef struct {
    double* velocity;  // Velocity vector
    double* gradient;  // Gradient vector
    double* position;  // Reference to Solution.position for sorting
} EVO_Particle;

// EVO Algorithm Phases
void evo_initialize_particles(Optimizer *opt, EVO_Particle *particles);
void evaluate_fitness_evo(Optimizer *opt, double (*objective_function)(double *));
void compute_gradient(Optimizer *opt, EVO_Particle *particles, double (*objective_function)(double *));
void update_velocity_and_position(Optimizer *opt, EVO_Particle *particles);
void free_evo_particles(EVO_Particle *particles, int population_size);

// Main Optimization Function
void EVO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // EVO_H
