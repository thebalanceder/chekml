#ifndef GWCA_H
#define GWCA_H

#include "generaloptimizer.h"

// GWCA-specific function prototypes
void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function);
int compare_fitness(const void *a, const void *b);
#endif // GWCA_H

