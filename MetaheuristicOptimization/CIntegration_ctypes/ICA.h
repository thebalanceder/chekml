#ifndef ICA_H
#define ICA_H

#include "generaloptimizer.h"

#define NUM_IMPERIALISTS 8
#define NUM_COUNTRIES 100
#define ZETA 0.1
#define ASSIMILATION_COEFF 2.0
#define REVOLUTION_RATE 0.3
#define DAMP_RATIO 0.99
#define UNITING_THRESHOLD 0.02
#define MAX_DECADES 500

typedef struct {
    double (*objective_function)(double *);
    int stop_if_single_empire;
    double revolution_rate;
} ICAParams;

typedef struct {
    double *imperialist_position;
    double imperialist_cost;
    double **colonies_position;
    double *colonies_cost;
    int num_colonies;
    double total_cost;
} Empire;

// Additional data for ICA optimization
typedef struct {
    int fes; // Function evaluations counter
} ICAOptimizerData;

void create_initial_empires(Optimizer *opt, ICAParams *params, Empire **empires, int *num_empires, ICAOptimizerData *data);
void assimilate_colonies(Optimizer *opt, Empire *empire);
void revolve_colonies(Optimizer *opt, Empire *empire, double revolution_rate);
void possess_empire(Optimizer *opt, ICAParams *params, Empire *empire, ICAOptimizerData *data);
void unite_similar_empires(Optimizer *opt, Empire **empires, int *num_empires);
void imperialistic_competition(Optimizer *opt, Empire **empires, int *num_empires);
void free_empires(Empire *empires, int num_empires, int dim);
void ICA_optimize(Optimizer *opt, double (*objective_function)(double *));
void ICA_optimize_wrapper(void *opt_void, double (*objective_function)(double *));

#endif
