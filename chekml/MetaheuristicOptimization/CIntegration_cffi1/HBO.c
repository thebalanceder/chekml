#include "HBO.h"
#include "generaloptimizer.h"
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

// 🌀 Fast RNG using xorshift
static uint32_t rng_state = 1;
static inline uint32_t fast_rand_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static inline double fast_rand_double(double min, double max) {
    return min + (max - min) * (fast_rand_u32() / (double)UINT32_MAX);
}

static inline void enforce_bounds(double *position, const double *bounds, int dim) {
    for (int j = 0; j < dim; ++j) {
        double x = position[j];
        double min = bounds[2 * j];
        double max = bounds[2 * j + 1];
        position[j] = (x < min) ? min : (x > max) ? max : x;
    }
}

static void initialize_population(Optimizer *opt) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    const double *bounds = opt->bounds;

    // Parallel population initialization
    #pragma omp parallel for
    for (int i = 0; i < pop_size; ++i) {
        Solution *ind = &opt->population[i];
        double *pos = ind->position;
        for (int j = 0; j < dim; ++j) {
            double lb = bounds[2 * j];
            double ub = bounds[2 * j + 1];
            pos[j] = fast_rand_double(lb, ub);
        }
        ind->fitness = INFINITY;
    }
}

void HBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    rng_state = (uint32_t)time(NULL);
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    const double *bounds = opt->bounds;

    initialize_population(opt);

    // Pre-allocate arrays for trial and heap candidate to avoid malloc inside loop
    double *trial = (double *)_mm_malloc(sizeof(double) * dim, 32);  // 32-byte alignment for SIMD
    double *heap_candidate = (double *)_mm_malloc(sizeof(double) * dim, 32);

    // Parallel population fitness evaluation and initialization
    #pragma omp parallel for
    for (int i = 0; i < pop_size; ++i) {
        Solution *ind = &opt->population[i];
        ind->fitness = objective_function(ind->position);

        if (ind->fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = ind->fitness;
            for (int j = 0; j < dim; ++j)
                opt->best_solution.position[j] = ind->position[j];
        }
    }

    for (int iter = 0; iter < opt->max_iter; ++iter) {
        #pragma omp parallel for
        for (int i = 0; i < pop_size; ++i) {
            Solution *ind = &opt->population[i];
            double *pos = ind->position;

            // Fast unique indices (RNG)
            int a, b, c;
            do {
                a = fast_rand_u32() % pop_size;
                b = fast_rand_u32() % pop_size;
                c = fast_rand_u32() % pop_size;
            } while (a == b || b == c || a == c);

            double *pa = opt->population[a].position;
            double *pb = opt->population[b].position;
            double *pc = opt->population[c].position;

            double r1 = fast_rand_double(0.0, 1.0);

            // SIMD optimization: Vectorized operation using AVX2 for position updates
            #pragma omp simd
            for (int j = 0; j < dim; ++j) {
                heap_candidate[j] = pa[j] + r1 * (pb[j] - pc[j]);
                trial[j] = pos[j];  // Initialize trial with the current individual position
            }

            // Randomly mutate a dimension
            int idx = fast_rand_u32() % dim;
            trial[idx] = heap_candidate[idx];

            enforce_bounds(trial, bounds, dim);
            double trial_fitness = objective_function(trial);

            // Update the individual if trial fitness is better
            if (trial_fitness < ind->fitness) {
                ind->fitness = trial_fitness;
                #pragma omp simd
                for (int j = 0; j < dim; ++j) {
                    pos[j] = trial[j];
                }

                // Check if the new fitness is the best solution
                if (trial_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = trial_fitness;
                    #pragma omp simd
                    for (int j = 0; j < dim; ++j) {
                        opt->best_solution.position[j] = trial[j];
                    }
                }
            }
        }
    }

    _mm_free(trial);
    _mm_free(heap_candidate);
}

