#include "SPO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>  // Include for SIMD

#define MAX_ITER 100
#define PAINT_FACTOR 0.1
#define INF 1e30
#define MAX_DIM 128  // Adjust based on your maximum dimension size

// SIMD optimized random number generation (for illustrative purposes)
inline double rand_uniform(double min, double max) {
    // Generate random numbers using SIMD (will generate 4 random numbers at once)
    __m128 rand_vals = _mm_set_ps(rand() / (double)RAND_MAX, rand() / (double)RAND_MAX,
                                  rand() / (double)RAND_MAX, rand() / (double)RAND_MAX);
    __m128 range_vals = _mm_sub_ps(_mm_set1_ps(max - min), _mm_set1_ps(min));
    return _mm_cvtss_f32(_mm_add_ps(_mm_set1_ps(min), _mm_mul_ps(rand_vals, range_vals)));
}

// Optimized bounding function using SIMD (to vectorize updates)
static inline void bound(double* restrict position, const double* restrict LB, const double* restrict UB, int dim) {
    int i;
    #pragma omp parallel for
	for (int i = 0; i < dim; i += 2) {
		__m128d pos = _mm_load_pd(&position[i]);
		__m128d lb = _mm_load_pd(&LB[i]);
		__m128d ub = _mm_load_pd(&UB[i]);

		__m128d min_vals = _mm_min_pd(pos, ub);
		__m128d final_vals = _mm_max_pd(min_vals, lb);
		_mm_store_pd(&position[i], final_vals);
	}
}

// Compare for qsort (no SIMD, simple comparison)
int compare_fitness(const void* a, const void* b) {
    double f1 = ((Solution*)a)->fitness;
    double f2 = ((Solution*)b)->fitness;
    return (f1 > f2) - (f1 < f2);
}

// SIMD optimized evaluation function using parallelism (fitness evaluation)
void evaluate_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Update population with SIMD optimizations (vectorization of position updates)
void update_population_spo(Optimizer* opt, ObjectiveFunction objective_function) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    int N1 = pop_size / 3, N2 = pop_size / 3;
    int N3 = pop_size - N1 - N2;

    qsort(opt->population, pop_size, sizeof(Solution), compare_fitness);

    Solution* Group1 = opt->population;
    Solution* Group2 = &opt->population[N1];
    Solution* Group3 = &opt->population[N1 + N2];

    const double* restrict bounds = opt->bounds;
    double LB[dim], UB[dim];
    for (int d = 0; d < dim; d++) {
        LB[d] = bounds[2 * d];
        UB[d] = bounds[2 * d + 1];
    }

    static double temp_positions[4][MAX_DIM];  // Adjust MAX_DIM based on your dimension size

    #pragma omp parallel for
    for (int ind = 0; ind < pop_size; ind++) {
        double* current = opt->population[ind].position;

        // SIMD-based Complement Combination using random positions
        int id1 = rand() % N1, id2 = rand() % N3;
        for (int d = 0; d < dim; d += 4) {
		__m128d current_vals = _mm_load_pd(&current[d]);
		__m128d group1_vals = _mm_load_pd(&Group1[id1].position[d]);
		__m128d group3_vals = _mm_load_pd(&Group3[id2].position[d]);

		__m128d rand_vals = _mm_set_pd(rand_uniform(0, 1), rand_uniform(0, 1));

		__m128d diff = _mm_sub_pd(group1_vals, group3_vals);
		__m128d scaled_diff = _mm_mul_pd(rand_vals, diff);
		__m128d temp_vals = _mm_add_pd(current_vals, scaled_diff);

		_mm_store_pd(&temp_positions[0][d], temp_vals);
        }

        // Process other combinations similarly (Analog, Triangle, Rectangle)

        // Apply bounds and calculate fitness for each combination
        for (int i = 0; i < 4; i++) {
            bound(temp_positions[i], LB, UB, dim);
            double fitness = objective_function(temp_positions[i]);
            if (fitness < opt->population[ind].fitness) {
                memcpy(current, temp_positions[i], sizeof(double) * dim);
                opt->population[ind].fitness = fitness;
            }
        }
    }
}

// Main SPO optimization loop with SIMD and parallelism
void SPO_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int pop_size = opt->population_size;
    int dim = opt->dim;

    for (int i = 0; i < pop_size; i++) {
        opt->population[i].position = (double*)aligned_alloc(32, sizeof(double) * dim);
        for (int d = 0; d < dim; d++) {
            opt->population[i].position[d] = rand_uniform(opt->bounds[2 * d], opt->bounds[2 * d + 1]);
        }
        opt->population[i].fitness = INF;
    }

    opt->best_solution.fitness = INF;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        evaluate_population_spo(opt, objective_function);
        update_population_spo(opt, objective_function);

        for (int i = 0; i < pop_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                memcpy(opt->best_solution.position, opt->population[i].position, sizeof(double) * dim);
                opt->best_solution.fitness = opt->population[i].fitness;
            }
        }

        printf("Iteration %3d: Best Fitness = %.10f\n", iter, opt->best_solution.fitness);
    }
}
