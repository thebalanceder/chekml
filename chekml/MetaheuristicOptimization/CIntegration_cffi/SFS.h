#ifndef SFS_H
#define SFS_H

#include "generaloptimizer.h"

// Structure for problem-specific parameters
typedef struct {
    int Start_Point;
    int Ndim;
    int Maximum_Generation;
    int Maximum_Diffusion;
    double Walk;
    double* Lband;
    double* Uband;
    double (*Function_Name)(const double*);  // Function pointer for objective function
} SFS_Params;

// Stochastic Fractal Search (SFS) optimizer
void SFS_optimize(Optimizer* opt, double (*obj_func)(const double*));

#endif
