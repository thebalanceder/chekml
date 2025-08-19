#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "SFS.h"

// Function to check if a point is within bounds
void Bound_Checking(double* point, double* Lband, double* Uband, int dim) {
    for (int i = 0; i < dim; i++) {
        if (point[i] < Lband[i]) point[i] = Lband[i];
        if (point[i] > Uband[i]) point[i] = Uband[i];
    }
}

// Diffusion process that mimics random walks
void Diffusion_Process(double* point, SFS_Params* S, int g, double* BestPoint, double* new_point, double* fitness, double (*obj_func)(const double*)) {
    int NumDiffusion = S->Maximum_Diffusion;
    for (int i = 0; i < NumDiffusion; i++) {
        double rand_val = (double)rand() / RAND_MAX;
        if (rand_val < S->Walk) {
            // Gaussian walk towards BestPoint
            for (int j = 0; j < S->Ndim; j++) {
                new_point[j] = point[j] + (rand_val * BestPoint[j] - (1 - rand_val) * point[j]);
            }
        } else {
            // Gaussian walk from current point
            for (int j = 0; j < S->Ndim; j++) {
                new_point[j] = point[j] + (rand_val * (BestPoint[j] - point[j]));
            }
        }
        Bound_Checking(new_point, S->Lband, S->Uband, S->Ndim);
        *fitness = obj_func(new_point);  // Call the objective function
    }
}

// The core optimization function
void SFS_optimize(Optimizer* opt, double (*obj_func)(const double*)) {
    if (!opt || !opt->population || !opt->best_solution.position || !opt->bounds || !obj_func) {
        fprintf(stderr, "❌ SFS_optimize received null pointer(s).\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;
    int max_iter = opt->max_iter;
    double* temp = (double*)malloc(dim * sizeof(double));

    SFS_Params S;
    S.Start_Point = pop_size;
    S.Ndim = dim;
    S.Maximum_Generation = max_iter;
    S.Maximum_Diffusion = 10;  // Example value
    S.Walk = 0.5;  // Example probability
    S.Lband = opt->bounds;
    S.Uband = opt->bounds + dim;
    S.Function_Name = obj_func;

    // Evaluate initial fitness
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = obj_func(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = opt->population[i].position[d];
            }
        }
    }

    // Main optimization loop
    for (int g = 0; g < max_iter; g++) {
        double* new_point = (double*)malloc(dim * sizeof(double));
        double* fitness_values = (double*)malloc(pop_size * sizeof(double));

        for (int i = 0; i < pop_size; i++) {
            Diffusion_Process(opt->population[i].position, &S, g, opt->best_solution.position, new_point, &fitness_values[i], obj_func);
            if (fitness_values[i] < opt->population[i].fitness) {
                opt->population[i].fitness = fitness_values[i];
                for (int d = 0; d < dim; d++) {
                    opt->population[i].position[d] = new_point[d];
                }
            }
        }

        // Update best solution
        double best_fit = DBL_MAX;
        for (int i = 0; i < pop_size; i++) {
            if (opt->population[i].fitness < best_fit) {
                best_fit = opt->population[i].fitness;
                for (int d = 0; d < dim; d++) {
                    opt->best_solution.position[d] = opt->population[i].position[d];
                }
            }
        }
        opt->best_solution.fitness = best_fit;

        free(new_point);
        free(fitness_values);
    }

    free(temp);
}
