#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <immintrin.h> // For SIMD operations
#include "SFS.h"

// Predefined random number generator (optional - faster)
static inline double rand_double() {
    return (double)rand() / RAND_MAX;
}

// Function to check if a point is within bounds
void Bound_Checking(double* point, double* Lband, double* Uband, int dim) {
    for (int i = 0; i < dim; i++) {
        if (point[i] < Lband[i]) point[i] = Lband[i];
        if (point[i] > Uband[i]) point[i] = Uband[i];
    }
}

// Diffusion process that mimics random walks, optimized with SIMD for vectorized operations
void Diffusion_Process(double* point, SFS_Params* S, int g, double* BestPoint, double* new_point, double* fitness, double (*obj_func)(const double*)) {
    int NumDiffusion = S->Maximum_Diffusion;
    for (int i = 0; i < NumDiffusion; i++) {
        double rand_val = rand_double();
        if (rand_val < S->Walk) {
            // Gaussian walk towards BestPoint using SIMD for vectorization
            for (int j = 0; j < S->Ndim; j += 4) {
                // Load 4 elements at a time using SIMD
                __m256d point_vec = _mm256_loadu_pd(&point[j]);
                __m256d best_vec = _mm256_loadu_pd(&BestPoint[j]);
                __m256d diff_vec = _mm256_sub_pd(best_vec, point_vec);

                // Apply the random walk update
                __m256d rand_vec = _mm256_set1_pd(rand_val);
                __m256d new_point_vec = _mm256_add_pd(point_vec, _mm256_mul_pd(rand_vec, diff_vec));
                _mm256_storeu_pd(&new_point[j], new_point_vec);
            }
        } else {
            // Gaussian walk from current point using SIMD for vectorization
            for (int j = 0; j < S->Ndim; j += 4) {
                // Load 4 elements at a time using SIMD
                __m256d point_vec = _mm256_loadu_pd(&point[j]);
                __m256d best_vec = _mm256_loadu_pd(&BestPoint[j]);
                __m256d diff_vec = _mm256_sub_pd(best_vec, point_vec);

                // Apply the random walk update
                __m256d rand_vec = _mm256_set1_pd(rand_val);
                __m256d new_point_vec = _mm256_add_pd(point_vec, _mm256_mul_pd(rand_vec, diff_vec));
                _mm256_storeu_pd(&new_point[j], new_point_vec);
            }
        }

        // Handle the remaining dimensions that cannot be processed in groups of 4
        for (int j = (S->Ndim / 4) * 4; j < S->Ndim; ++j) {
            new_point[j] = point[j] + (rand_val * (BestPoint[j] - point[j]));
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

    SFS_Params S;
    S.Start_Point = pop_size;
    S.Ndim = dim;
    S.Maximum_Generation = max_iter;
    S.Maximum_Diffusion = 10;  // Example value
    S.Walk = 0.5;  // Example probability
    S.Lband = opt->bounds;
    S.Uband = opt->bounds + dim;
    S.Function_Name = obj_func;

    // Preallocate fitness array to avoid frequent malloc/free
    double* fitness_values = (double*)malloc(pop_size * sizeof(double));
    if (!fitness_values) {
        fprintf(stderr, "❌ Memory allocation failed for fitness values.\n");
        return;
    }

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
        double* new_point = (double*)calloc(dim, sizeof(double));  // Use calloc to ensure initialization to zero
        if (!new_point) {
            fprintf(stderr, "❌ Memory allocation failed for new_point.\n");
            free(fitness_values);
            return;
        }

        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            Diffusion_Process(opt->population[i].position, &S, g, opt->best_solution.position, new_point, &fitness_values[i], obj_func);
            if (fitness_values[i] < opt->population[i].fitness) {
                opt->population[i].fitness = fitness_values[i];
                for (int d = 0; d < dim; d++) {
                    opt->population[i].position[d] = new_point[d];
                }
            }
        }

        // Update best solution in parallel
        double best_fit = DBL_MAX;
        #pragma omp parallel for reduction(min:best_fit)
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
    }

    free(fitness_values);
}
