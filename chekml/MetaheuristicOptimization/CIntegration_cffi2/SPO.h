#ifndef SPO_H
#define SPO_H

#include "generaloptimizer.h"

// SPO-specific function prototypes
void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void enforce_bound_constraints(Optimizer* opt);
double rand_uniform(double min, double max);  // Random number generator

void evaluate_population_spo(Optimizer* opt, ObjectiveFunction objective_function);
#endif // SPO_H
