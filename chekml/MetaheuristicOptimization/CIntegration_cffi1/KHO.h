#ifndef KHO_H
#define KHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

#define VF 0.02
#define DMAX 0.005
#define NMAX 0.01
#define CROSSOVER_RATE 0.8
#define CROSSOVER_SCALE 0.2
#define INERTIA_MIN 0.1
#define INERTIA_MAX 0.8
#define NEIGHBOR_LIMIT 4
#define SENSE_DISTANCE_FACTOR 5.0
#define KHO_MAX_DIM 100
#define KHO_MAX_POP 1000
#define KHO_MAX_ITER 10000
#define ALIGN 32

typedef struct {
    double fitness;
    double position[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
} KHO_History;

typedef struct {
    double best_position[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double best_fitness;
    double N[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double F[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double D[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
} KHO_Solution;

typedef struct {
    double Rgb[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double RR[KHO_MAX_DIM * KHO_MAX_POP] __attribute__((aligned(ALIGN)));
    double R[KHO_MAX_POP] __attribute__((aligned(ALIGN)));
    double Rf[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double Rib[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double Xf[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double Sf[KHO_MAX_DIM] __attribute__((aligned(ALIGN)));
    double K[KHO_MAX_POP] __attribute__((aligned(ALIGN)));
    double Kib[KHO_MAX_POP] __attribute__((aligned(ALIGN)));
} KHO_Buffers;

typedef struct {
    Optimizer* base;
    int current_iter;
    double Dt;
    int crossover_flag;
    double inv_max_iter;
    KHO_History history[KHO_MAX_ITER];
    KHO_Solution solutions[KHO_MAX_POP];
    KHO_Buffers buffers;
} KHO_Optimizer;

double rand_double(double min, double max);

void initialize_krill_positions(KHO_Optimizer *opt);
void evaluate_krill_positions(KHO_Optimizer *opt, double (*objective_function)(double *));
void movement_induced_phase(KHO_Optimizer *opt, double w, double Kw_Kgb);
void foraging_motion_phase(KHO_Optimizer *opt, double w, double Kf, double Kw_Kgb);
void physical_diffusion_phase(KHO_Optimizer *opt, int iteration, double Kw_Kgb);
void crossover_phase(KHO_Optimizer *opt, double Kw_Kgb);
void kho_update_positions(KHO_Optimizer *opt);
void enforce_kho_bounds(KHO_Optimizer *opt, double *position, double *best);

void KHO_optimize(Optimizer *base, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif
