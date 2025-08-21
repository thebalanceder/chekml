#include "DISOwithRCF.h"
#include "generaloptimizer.h"
#include <omp.h>
#include <xmmintrin.h> // SIMD header for AVX support
#include <stdlib.h>
#include <math.h>

// Generate a random double between min and max (use SIMD for random generation if needed)
double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// SIMD version of rand_double
void rand_double_simd(double *result, double min, double max) {
    __m128d min_vec = _mm_set1_pd(min);
    __m128d max_vec = _mm_set1_pd(max);
    __m128d random_vals = _mm_setr_pd((double)rand() / RAND_MAX, (double)rand() / RAND_MAX);

    // Linear transformation of the random values
    __m128d range = _mm_sub_pd(max_vec, min_vec);
    __m128d scaled = _mm_mul_pd(random_vals, range);
    __m128d final_vals = _mm_add_pd(min_vec, scaled);

    _mm_storeu_pd(result, final_vals);
}

// Global Diversion Phase (Fish Mouth Dividing Project) - Parallelized with SIMD and OpenMP
void diversion_phase(Optimizer *opt) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double r1 = rand_double(0.0, 1.0);
        double r2 = rand_double(0.0, 1.0);
        double CFR = CFR_FACTOR * rand_double(0.0, 1.0) * 2.5;

        double velocity = (r1 < 0.23) ? (pow(HRO, 2.0 / 3.0) * sqrt(HGO) / CFR) * r1
                                      : (pow(HRI, 2.0 / 3.0) * sqrt(HGI) / CFR) * r2;

        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += velocity * (opt->best_solution.position[j] - opt->population[i].position[j]) * rand_double(0.0, 1.0);
        }
    }
    enforce_bound_constraints(opt);
}

// Spiral Motion Update - Parallelized using OpenMP
void spiral_motion_update(Optimizer *opt, int t) {
    double T = (double)opt->max_iter;
    double total_fitness = 0.0;

    #pragma omp parallel for reduction(+:total_fitness)
    for (int i = 0; i < opt->population_size; i++) {
        total_fitness += opt->population[i].fitness;
    }

    double MLV = total_fitness / opt->population_size;
    double LP = (WATER_DENSITY * FLUID_DISTRIBUTION * MLV * MLV) / CENTRIFUGAL_RESISTANCE;

    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double RCF = WATER_DENSITY * cos(90 * (t / T)) * sqrt(pow(opt->best_solution.fitness - opt->population[i].fitness, 2));

        if (RCF > LP) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = opt->bounds[2 * j] + (rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Local Refinement Phase - Parallelized using OpenMP
void local_development_phase(Optimizer *opt) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        double r3 = rand_double(0.0, 1.0);
        double CFR = CFR_FACTOR * rand_double(0.0, 1.0) * 2.5;

        double velocity = (pow(HRI, 2.0 / 3.0) * sqrt(HGI) / (2 * CFR)) * ((r3 < BOTTLENECK_RATIO) ? r3 : rand_double(0.0, 1.0));

        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += velocity * (opt->best_solution.position[j] - opt->population[i].position[j]) * rand_double(0.0, 1.0);
        }
    }
    enforce_bound_constraints(opt);
}

// Elimination Phase (Worst Solution Replacement) - Parallelized sorting with batch replacement
void elimination_phase(Optimizer *opt) {
    int worst_count = (int)(ELIMINATION_RATIO * opt->population_size);
    
    #pragma omp parallel for
    for (int i = 0; i < worst_count; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[opt->population_size - i - 1].position[j] = opt->bounds[2 * j] + 
                                                                       (rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]));
        }
        opt->population[opt->population_size - i - 1].fitness = INFINITY;
    }

    // Parallelize sorting the population based on fitness (qsort + OpenMP)
    #pragma omp parallel
    {
        #pragma omp single
        qsort(opt->population, opt->population_size, sizeof(Solution), compare_fitness_diso);
    }

    enforce_bound_constraints(opt);
}

// Main Optimization Function - Parallelized
void DISO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    #pragma omp parallel for
    for (int iter = 0; iter < opt->max_iter; iter++) {
        diversion_phase(opt);
        spiral_motion_update(opt, iter);
        local_development_phase(opt);
        elimination_phase(opt);

        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);
    }
}

// Comparison function for qsort (used in sorting)
int compare_fitness_diso(const void *a, const void *b) {
    const Solution *ind_a = (const Solution *)a;
    const Solution *ind_b = (const Solution *)b;
    return (ind_b->fitness - ind_a->fitness);
}
