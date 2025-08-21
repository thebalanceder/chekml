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

// Function prototypes
double rand_double(double min, double max);
void initialize_population_fho(Optimizer *opt);
double euclidean_distance(double *a, double *b, int dim);
void assign_prey_to_firehawks(Optimizer *opt, int num_firehawks, int *prey_counts, int **prey_indices);
void update_firehawk_position(Optimizer *opt, double *fh, double *other_fh, double *new_pos);
void update_prey_position(Optimizer *opt, double *prey, double *firehawk, double *local_safe_point, double *global_safe_point, double *pos1, double *pos2);
void FHO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FHO_H
