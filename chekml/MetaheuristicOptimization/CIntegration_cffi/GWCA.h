#ifndef GWCA_H
#define GWCA_H

#include "generaloptimizer.h"

// GWCA-specific constants
#define G 9.8      // Gravitational constant (m/s^2)
#define M 3        // Some constant for the method
#define E 0.1      // Some constant for the method
#define P 9        // Some constant for the method
#define Q 6        // Some constant for the method
#define CMAX 20    // Max value for constant C
#define CMIN 10    // Min value for constant C

// GWCA-specific function prototypes
void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function);
int compare_fitness_gwca(const void *a, const void *b);
#endif // GWCA_H

