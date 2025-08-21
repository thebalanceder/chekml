#include "MA.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics

#define aligned_alloc_double(n) ((double*)aligned_alloc(32, (n) * sizeof(double)))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline double rand_normal(double stddev) {
    double u1 = rand_uniform(0.0, 1.0);
    double u2 = rand_uniform(0.0, 1.0);
    return stddev * sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

static inline void clamp_simd(double* vec, int dim, const double* bounds) {
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        __m256d v = _mm256_loadu_pd(&vec[i]);
        __m256d minb = _mm256_loadu_pd(&bounds[2 * i]);
        __m256d maxb = _mm256_loadu_pd(&bounds[2 * i + 1]);

        v = _mm256_max_pd(minb, _mm256_min_pd(maxb, v));
        _mm256_storeu_pd(&vec[i], v);
    }
    // Fallback for remaining dimensions
    for (; i < dim; i++) {
        vec[i] = MAX(bounds[2 * i], MIN(bounds[2 * i + 1], vec[i]));
    }
}


static void hill_climb_simd(double* sol, int dim, const double* bounds, ObjectiveFunction f) {
    double* candidate = (double*)aligned_alloc_double(sizeof(double) * dim);
    double best_fit = f(sol);

    for (int it = 0; it < MA_LOCAL_SEARCH_ITERS; it++) {
        int i = 0;
        for (; i <= dim - 4; i += 4) {
            double noise[4] = {
                rand_normal(MA_MUTATION_STDDEV),
                rand_normal(MA_MUTATION_STDDEV),
                rand_normal(MA_MUTATION_STDDEV),
                rand_normal(MA_MUTATION_STDDEV)
            };
            __m256d orig = _mm256_loadu_pd(&sol[i]);
            __m256d randv = _mm256_loadu_pd(noise);
            __m256d cand = _mm256_add_pd(orig, randv);

            __m256d minb = _mm256_loadu_pd(&bounds[2 * i]);
            __m256d maxb = _mm256_loadu_pd(&bounds[2 * i + 1]);
            cand = _mm256_max_pd(minb, _mm256_min_pd(maxb, cand));
            _mm256_storeu_pd(&candidate[i], cand);
        }

        for (; i < dim; i++) {
            candidate[i] = sol[i] + rand_normal(MA_MUTATION_STDDEV);
            candidate[i] = MAX(bounds[2 * i], MIN(bounds[2 * i + 1], candidate[i]));
        }

        double fit = f(candidate);
        if (fit < best_fit) {
            memcpy(sol, candidate, sizeof(double) * dim);
            best_fit = fit;
        }
    }
    free(candidate);
}

static void crossover(const double* p1, const double* p2, double* c1, double* c2, int dim) {
    int cx = rand() % dim;
    for (int i = 0; i < dim; i++) {
        c1[i] = (i <= cx) ? p1[i] : p2[i];
        c2[i] = (i <= cx) ? p2[i] : p1[i];
    }
}

static void mutate_simd(double* sol, int dim, const double* bounds) {
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        double noise[4] = {
            rand_normal(MA_MUTATION_STDDEV),
            rand_normal(MA_MUTATION_STDDEV),
            rand_normal(MA_MUTATION_STDDEV),
            rand_normal(MA_MUTATION_STDDEV)
        };
        __m256d v = _mm256_loadu_pd(&sol[i]);
        __m256d n = _mm256_loadu_pd(noise);
        v = _mm256_add_pd(v, n);

        __m256d minb = _mm256_loadu_pd(&bounds[2 * i]);
        __m256d maxb = _mm256_loadu_pd(&bounds[2 * i + 1]);
        v = _mm256_max_pd(minb, _mm256_min_pd(maxb, v));
        _mm256_storeu_pd(&sol[i], v);
    }
    for (; i < dim; i++) {
        sol[i] += rand_normal(MA_MUTATION_STDDEV);
        sol[i] = MAX(bounds[2 * i], MIN(bounds[2 * i + 1], sol[i]));
    }
}


void MA_optimize(Optimizer* opt, ObjectiveFunction f) {
    int dim = opt->dim, pop = opt->population_size, max_iter = opt->max_iter;
    int total = pop * dim;
    double* population = (double*)aligned_alloc_double(sizeof(double) * total);
    double* fitness = (double*)aligned_alloc_double(sizeof(double) * pop);

    // Bounds copy
    double* bounds = (double*)aligned_alloc_double(sizeof(double) * dim * 2);
    memcpy(bounds, opt->bounds, sizeof(double) * dim * 2);

    // Initial population
    #pragma omp parallel for
    for (int i = 0; i < pop; i++) {
        for (int d = 0; d < dim; d++)
            population[i * dim + d] = rand_uniform(bounds[2 * d], bounds[2 * d + 1]);
        fitness[i] = f(&population[i * dim]);
    }

    double* buffer = (double*)aligned_alloc_double(sizeof(double) * dim);
    double* children = (double*)aligned_alloc_double(sizeof(double) * pop * dim);  // worst-case for space reuse

    for (int iter = 0; iter < max_iter; iter++) {
        int ls_count = (int)(MA_LOCAL_SEARCH_RATE * pop);

        // Hill Climbing in Parallel
        #pragma omp parallel for private(buffer)
        for (int i = 0; i < ls_count; i++) {
            int idx = rand() % pop;
            hill_climb_simd(&population[idx * dim], dim, bounds, f);
            fitness[idx] = f(&population[idx * dim]);
        }

        // Select top individuals by sorting indices
        int* indices = (int*)aligned_alloc_double(sizeof(int) * pop);
        for (int i = 0; i < pop; i++) indices[i] = i;
        for (int i = 0; i < pop - 1; i++) {
            for (int j = i + 1; j < pop; j++) {
                if (fitness[indices[i]] > fitness[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        // Crossover and Mutation
        int num_children = ((int)(MA_CROSSOVER_RATE * pop)) & ~1;  // even
        for (int i = 0; i < num_children; i += 2) {
            int p1 = indices[i], p2 = indices[i + 1];
            double* c1 = &children[i * dim];
            double* c2 = &children[(i + 1) * dim];
            crossover(&population[p1 * dim], &population[p2 * dim], c1, c2, dim);
            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate_simd(c1, dim, bounds);
            if (rand_uniform(0.0, 1.0) < MA_MUTATION_RATE) mutate_simd(c2, dim, bounds);
        }

        // Replace worst with children
        for (int i = 0; i < num_children; i++) {
            int idx = indices[pop - 1 - i];
            memcpy(&population[idx * dim], &children[i * dim], sizeof(double) * dim);
            fitness[idx] = f(&population[idx * dim]);
        }

        // Best solution tracking
        int best_idx = 0;
        for (int i = 1; i < pop; i++)
            if (fitness[i] < fitness[best_idx]) best_idx = i;

        if (fitness[best_idx] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[best_idx];
            memcpy(opt->best_solution.position, &population[best_idx * dim], dim * sizeof(double));
        }

        printf("MA | Iteration %4d -> Best Fitness = %.16f\n", iter + 1, opt->best_solution.fitness);
        free(indices);
    }

    free(population);
    free(fitness);
    free(bounds);
    free(children);
}

