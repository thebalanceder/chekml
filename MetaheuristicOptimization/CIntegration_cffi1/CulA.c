#include "CulA.h"
#include <stdlib.h>  // ✅ For malloc(), qsort()
#include <time.h>    // ✅ For initial seed
#include <math.h>    // ✅ For mathematical functions
#include <stdio.h>   // ✅ For error logging
#include <stdint.h>  // ✅ For uint64_t
#include <omp.h>     // ✅ For parallelization
#include <immintrin.h> // ✅ For SIMD intrinsics (AVX)

// 🎲 Static state for xorshift_normal
static int has_spare = 0;
static double spare = 0.0;

// 🎲 Comparison function for qsort
typedef struct {
    double fitness;
    int index;
} Individual;

int compare_individuals(const void *a, const void *b) {
    double diff = ((Individual *)a)->fitness - ((Individual *)b)->fitness;
    return (diff > 0) - (diff < 0);
}

// 🌍 Initialize Culture Components
void initialize_culture(Optimizer *opt, Culture *culture) {
    if (!opt || !culture) {
        fprintf(stderr, "🚫 Error: Null optimizer or culture\n");
        return;
    }

    int dim = opt->dim;
    double *mem_block = (double *)malloc(5 * dim * sizeof(double));
    if (!mem_block) {
        fprintf(stderr, "🚫 Error: Memory allocation failed\n");
        return;
    }

    culture->situational.position = (double *)malloc(dim * sizeof(double));
    if (!culture->situational.position) {
        fprintf(stderr, "🚫 Error: Memory allocation failed for situational.position\n");
        free(mem_block);
        return;
    }

    culture->normative.min = mem_block;
    culture->normative.max = mem_block + dim;
    culture->normative.L = mem_block + 2 * dim;
    culture->normative.U = mem_block + 3 * dim;
    culture->normative.size = mem_block + 4 * dim;

    culture->situational.cost = INFINITY;
    // Vectorized initialization with AVX
    int j = 0;
    for (; j <= dim - 4; j += 4) {
        _mm256_storeu_pd(&culture->situational.position[j], _mm256_setzero_pd());
        _mm256_storeu_pd(&culture->normative.min[j], _mm256_set1_pd(INFINITY));
        _mm256_storeu_pd(&culture->normative.max[j], _mm256_set1_pd(-INFINITY));
        _mm256_storeu_pd(&culture->normative.L[j], _mm256_set1_pd(INFINITY));
        _mm256_storeu_pd(&culture->normative.U[j], _mm256_set1_pd(INFINITY));
        _mm256_storeu_pd(&culture->normative.size[j], _mm256_setzero_pd());
    }
    // Handle remaining elements
    #pragma omp parallel for
    for (int k = j; k < dim; k++) {
        culture->situational.position[k] = 0.0;
        culture->normative.min[k] = INFINITY;
        culture->normative.max[k] = -INFINITY;
        culture->normative.L[k] = INFINITY;
        culture->normative.U[k] = INFINITY;
        culture->normative.size[k] = 0.0;
    }
}

// 🌍 Adjust Culture Based on Top n_accept Individuals
void adjust_culture(Optimizer *opt, Culture *culture, int n_accept) {
    if (!opt || !culture || !opt->population || n_accept <= 0) {
        fprintf(stderr, "🚫 Error: Invalid inputs in adjust_culture\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;

    Individual *individuals = (Individual *)malloc(pop_size * sizeof(Individual));
    if (!individuals) {
        fprintf(stderr, "🚫 Error: Memory allocation failed\n");
        return;
    }

    // Collect fitness values and indices
    #pragma omp parallel for
    for (int i = 0; i < pop_size; i++) {
        individuals[i].fitness = opt->population[i].fitness;
        individuals[i].index = i;
    }

    // Sort using qsort
    qsort(individuals, pop_size, sizeof(Individual), compare_individuals);

    // Update situational and normative knowledge
    for (int i = 0; i < n_accept && i < pop_size; i++) {
        int idx = individuals[i].index;
        double cost = individuals[i].fitness;

        // Situational component
        if (cost < culture->situational.cost) {
            culture->situational.cost = cost;
            // Vectorized copy with AVX
            int j = 0;
            for (; j <= dim - 4; j += 4) {
                _mm256_storeu_pd(&culture->situational.position[j],
                    _mm256_loadu_pd(&opt->population[idx].position[j]));
            }
            for (int k = j; k < dim; k++) {
                culture->situational.position[k] = opt->population[idx].position[k];
            }
        }

        // Normative component (vectorized)
        double *pos = opt->population[idx].position;
        int j = 0;
        __m256d cost_vec = _mm256_set1_pd(cost);
        for (; j <= dim - 4; j += 4) {
            __m256d pos_vec = _mm256_loadu_pd(&pos[j]);
            __m256d min_vec = _mm256_loadu_pd(&culture->normative.min[j]);
            __m256d max_vec = _mm256_loadu_pd(&culture->normative.max[j]);
            __m256d L_vec = _mm256_loadu_pd(&culture->normative.L[j]);
            __m256d U_vec = _mm256_loadu_pd(&culture->normative.U[j]);

            // Update min and L
            __m256d min_mask = _mm256_or_pd(
                _mm256_cmp_pd(pos_vec, min_vec, _CMP_LT_OQ),
                _mm256_cmp_pd(cost_vec, L_vec, _CMP_LT_OQ));
            min_vec = _mm256_blendv_pd(min_vec, pos_vec, min_mask);
            L_vec = _mm256_blendv_pd(L_vec, cost_vec, min_mask);

            // Update max and U
            __m256d max_mask = _mm256_or_pd(
                _mm256_cmp_pd(pos_vec, max_vec, _CMP_GT_OQ),
                _mm256_cmp_pd(cost_vec, U_vec, _CMP_LT_OQ));
            max_vec = _mm256_blendv_pd(max_vec, pos_vec, max_mask);
            U_vec = _mm256_blendv_pd(U_vec, cost_vec, max_mask);

            // Store results
            _mm256_storeu_pd(&culture->normative.min[j], min_vec);
            _mm256_storeu_pd(&culture->normative.max[j], max_vec);
            _mm256_storeu_pd(&culture->normative.L[j], L_vec);
            _mm256_storeu_pd(&culture->normative.U[j], U_vec);
            _mm256_storeu_pd(&culture->normative.size[j], _mm256_sub_pd(max_vec, min_vec));
        }
        // Handle remaining elements
        #pragma omp parallel for
        for (int k = j; k < dim; k++) {
            double p = pos[k];
            if (p < culture->normative.min[k] || cost < culture->normative.L[k]) {
                culture->normative.min[k] = p;
                culture->normative.L[k] = cost;
            }
            if (p > culture->normative.max[k] || cost < culture->normative.U[k]) {
                culture->normative.max[k] = p;
                culture->normative.U[k] = cost;
            }
            culture->normative.size[k] = culture->normative.max[k] - culture->normative.min[k];
        }
    }

    free(individuals);
}

// 🌍 Apply Cultural Influence to Update Population
void influence_culture(Optimizer *opt, Culture *culture) {
    if (!opt || !culture || !opt->population || !culture->situational.position) {
        fprintf(stderr, "🚫 Error: Invalid inputs in influence_culture\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;
    XorshiftState rng;
    rng.state = (uint64_t)time(NULL) ^ (uint64_t)opt;

    // Precompute alpha-scaled sizes
    double *sigma = (double *)malloc(dim * sizeof(double));
    int j = 0;
    for (; j <= dim - 4; j += 4) {
        __m256d size_vec = _mm256_loadu_pd(&culture->normative.size[j]);
        _mm256_storeu_pd(&sigma[j], _mm256_mul_pd(size_vec, _mm256_set1_pd(ALPHA_SCALING)));
    }
    for (int k = j; k < dim; k++) {
        sigma[k] = ALPHA_SCALING * culture->normative.size[k];
    }

    #pragma omp parallel private(rng)
    {
        rng.state = (uint64_t)time(NULL) ^ (uint64_t)omp_get_thread_num();
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j += 4) {
                // Process 4 dimensions at a time (AVX)
                if (j + 4 <= dim) {
                    __m256d pos_vec = _mm256_loadu_pd(&opt->population[i].position[j]);
                    __m256d sit_pos_vec = _mm256_loadu_pd(&culture->situational.position[j]);
                    __m256d sigma_vec = _mm256_loadu_pd(&sigma[j]);
                    __m256d dx_vec = _mm256_set_pd(
                        has_spare ? (has_spare = 0, spare) : (has_spare = 1, spare = xorshift_normal(&rng, 0.0, sigma[j + 3]), spare),
                        has_spare ? (has_spare = 0, spare) : (has_spare = 1, spare = xorshift_normal(&rng, 0.0, sigma[j + 2]), spare),
                        has_spare ? (has_spare = 0, spare) : (has_spare = 1, spare = xorshift_normal(&rng, 0.0, sigma[j + 1]), spare),
                        has_spare ? (has_spare = 0, spare) : (has_spare = 1, spare = xorshift_normal(&rng, 0.0, sigma[j]), spare)
                    );

                    // Conditional update: dx = fabs(dx) if pos < sit_pos, else -fabs(dx)
                    __m256d cmp_mask = _mm256_cmp_pd(pos_vec, sit_pos_vec, _CMP_LT_OQ);
                    __m256d abs_dx = _mm256_andnot_pd(_mm256_set1_pd(-0.0), dx_vec); // fabs
                    __m256d neg_abs_dx = _mm256_sub_pd(_mm256_setzero_pd(), abs_dx);
                    dx_vec = _mm256_blendv_pd(neg_abs_dx, abs_dx, cmp_mask);

                    // Update position
                    _mm256_storeu_pd(&opt->population[i].position[j], _mm256_add_pd(pos_vec, dx_vec));
                } else {
                    // Handle remaining dimensions
                    for (int k = j; k < dim; k++) {
                        double pos = opt->population[i].position[k];
                        double sit_pos = culture->situational.position[k];
                        double dx = has_spare ? (has_spare = 0, spare) : (has_spare = 1, spare = xorshift_normal(&rng, 0.0, sigma[k]), spare);
                        dx = (pos < sit_pos) ? fabs(dx) : -fabs(dx);
                        opt->population[i].position[k] = pos + dx;
                    }
                }
            }
        }
    }

    free(sigma);
    enforce_bound_constraints(opt);
}

// 🚀 Main Optimization Function
void CulA_optimize(void *opt_void, ObjectiveFunction objective_function) {
    if (!opt_void || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs to CulA_optimize\n");
        return;
    }

    Optimizer *opt = (Optimizer *)opt_void;
    if (!opt->population || !opt->bounds || opt->dim <= 0 || opt->population_size <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "🚫 Error: Invalid optimizer parameters\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;

    // Initialize RNG
    XorshiftState rng;
    rng.state = (uint64_t)time(NULL);

    // Initialize culture
    Culture culture = {0};
    initialize_culture(opt, &culture);
    if (!culture.situational.position) {
        fprintf(stderr, "🚫 Error: Culture initialization failed\n");
        return;
    }

    // Initialize population
    #pragma omp parallel private(rng)
    {
        rng.state = (uint64_t)time(NULL) ^ (uint64_t)omp_get_thread_num();
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                double min_bound = opt->bounds[2 * j];
                double max_bound = opt->bounds[2 * j + 1];
                opt->population[i].position[j] = xorshift_double(&rng, min_bound, max_bound);
            }
        }
    }

    // Evaluate initial population fitness
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Calculate number of accepted individuals
    int n_accept = (int)(ACCEPTANCE_RATIO * pop_size);
    if (n_accept < 1) n_accept = 1;

    // Initialize culture with initial population
    adjust_culture(opt, &culture, n_accept);

    // Set initial best solution
    if (!opt->best_solution.position) {
        fprintf(stderr, "🚫 Error: Null best_solution.position\n");
        free(culture.situational.position);
        free(culture.normative.min);
        return;
    }
    opt->best_solution.fitness = culture.situational.cost;
    // Vectorized copy
    int j = 0;
    for (; j <= dim - 4; j += 4) {
        _mm256_storeu_pd(&opt->best_solution.position[j],
            _mm256_loadu_pd(&culture.situational.position[j]));
    }
    for (int k = j; k < dim; k++) {
        opt->best_solution.position[k] = culture.situational.position[k];
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        influence_culture(opt, &culture);

        // Evaluate fitness in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < pop_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }

        adjust_culture(opt, &culture, n_accept);

        if (culture.situational.cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = culture.situational.cost;
            j = 0;
            for (; j <= dim - 4; j += 4) {
                _mm256_storeu_pd(&opt->best_solution.position[j],
                    _mm256_loadu_pd(&culture.situational.position[j]));
            }
            for (int k = j; k < dim; k++) {
                opt->best_solution.position[k] = culture.situational.position[k];
            }
        }
        printf("🔄 Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(culture.situational.position);
    free(culture.normative.min);
}
