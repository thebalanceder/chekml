#ifndef FHO_H
#define FHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MIN_FIREHAWKS 1
#define MAX_FIREHAWKS_RATIO 0.2  // Max firehawks = ceil(population_size * MAX_FIREHAWKS_RATIO)
#define IR_MIN 0.0
#define IR_MAX 1.0
#define CACHE_LINE_SIZE 64  // For memory alignment

// Function prototypes
double rand_double(double min, double max);
void initialize_population_fho(Optimizer *opt);
double euclidean_distance(double *a, double *b, int dim);
void assign_prey_to_firehawks(Optimizer *opt, int num_firehawks, int *prey_assignments, int *prey_counts);
void FHO_optimize(Optimizer *opt, double (*objective_function)(double *));

// Inline function definitions
inline void update_firehawk_position(Optimizer *opt, double *fh, double *other_fh, double *new_pos) {
    double ir1 = rand_double(IR_MIN, IR_MAX);
    double ir2 = rand_double(IR_MIN, IR_MAX);
    #pragma omp simd
    for (int i = 0; i < opt->dim; i++) {
        new_pos[i] = fh[i] + (ir1 * opt->best_solution.position[i] - ir2 * other_fh[i]);
        new_pos[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], new_pos[i]));
    }
}

inline void update_prey_position(Optimizer *opt, double *prey, double *firehawk, double *local_safe_point, 
                                double *global_safe_point, double *pos1, double *pos2) {
    double ir1 = rand_double(IR_MIN, IR_MAX);
    double ir2 = rand_double(IR_MIN, IR_MAX);
    #pragma omp simd
    for (int i = 0; i < opt->dim; i++) {
        pos1[i] = prey[i] + (ir1 * firehawk[i] - ir2 * local_safe_point[i]);
        pos1[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], pos1[i]));
    }

    ir1 = rand_double(IR_MIN, IR_MAX);
    ir2 = rand_double(IR_MIN, IR_MAX);
    int rand_fh_idx = rand() % opt->population_size;
    double *other_fh = opt->population[rand_fh_idx].position;
    #pragma omp simd
    for (int i = 0; i < opt->dim; i++) {
        pos2[i] = prey[i] + (ir1 * other_fh[i] - ir2 * global_safe_point[i]);
        pos2[i] = fmax(opt->bounds[2 * i], fmin(opt->bounds[2 * i + 1], pos2[i]));
    }
}

#ifdef __cplusplus
}
#endif

#endif // FHO_H
