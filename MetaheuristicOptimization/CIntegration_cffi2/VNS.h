#ifndef VNS_H
#define VNS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "generaloptimizer.h"

#define MUTATION_RATE 0.1

double rand_double_vns(double min, double max);

// VNS.h

void generate_neighbor_vns(const double * restrict current_solution,
                           double * restrict neighbor,
                           int dim,
                           const double * restrict bounds,
                           int neighborhood_size);

void VNS_optimize(Optimizer *opt, double (*objective_function)(double *), 
                  int max_iterations, const int *neighborhood_sizes, int num_neighborhoods);

#ifdef __cplusplus
}
#endif

#endif // VNS_H
