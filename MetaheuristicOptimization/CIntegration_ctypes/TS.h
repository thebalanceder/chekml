#ifndef TABU_SEARCH_H
#define TABU_SEARCH_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "generaloptimizer.h"

#define TABU_TENURE 10
#define NEIGHBORHOOD_SIZE 20
#define STEP_SIZE 0.1

typedef struct {
    double *moves;     // Linear buffer for storing move vectors
    int *tenures;      // Parallel buffer for tenures
    int capacity;
    int dim;
    int count;
} TabuList;

void init_tabu_list(TabuList *tabu, int capacity, int dim);
void free_tabu_list(TabuList *tabu);
int is_move_tabu(TabuList *tabu, double *move);
void add_tabu_move(TabuList *tabu, double *move);
void decrement_tabu_tenures(TabuList *tabu);

void generate_neighbor(double *neighbor, double *current, double *lower, double *upper, int dim);
void TS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TABU_SEARCH_H

