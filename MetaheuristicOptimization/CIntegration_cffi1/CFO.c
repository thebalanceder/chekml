/* CFO.c - Highly Optimized Implementation for Central Force Optimization */
/* Compile with: -O3 -ffast-math -march=native -mavx2 -std=c11 */
#include "CFO.h"
#include <immintrin.h>
#include <stdint.h>
#include <time.h>

/* Fast Xorshift PRNG */
static uint64_t xorshift_state = 1;

static inline uint64_t xorshift64(void) {
    uint64_t x = xorshift_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    xorshift_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double xorshift_double(void) {
    return (double)(xorshift64() >> 11) / (double)(1ULL << 53);
}

/* Initialize Population with SIMD and fast PRNG */
void initialize_population_cfo(Optimizer *restrict opt) {
    xorshift_state = (uint64_t)time(NULL); /* Seed once */
    const int dim = opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        int j = 0;
        /* Process 4 dimensions at a time with AVX2 */
        for (; j <= dim - 4; j += 4) {
            __m256d min = _mm256_loadu_pd(&opt->bounds[2 * j]);
            __m256d max = _mm256_loadu_pd(&opt->bounds[2 * j + 4]);
            __m256d range = _mm256_sub_pd(max, min);
            __m256d rand = _mm256_set_pd(xorshift_double(), xorshift_double(), xorshift_double(), xorshift_double());
            __m256d result = _mm256_fmadd_pd(range, rand, min);
            _mm256_storeu_pd(&pos[j], result);
        }
        /* Handle remaining dimensions */
        for (; j < dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            pos[j] = min + (max - min) * xorshift_double();
        }
        opt->population[i].fitness = INFINITY;
    }
}

/* Central Force Update with SIMD and loop unrolling */
void central_force_update(Optimizer *restrict opt) {
    const int dim = opt->dim;
    double center_of_mass[dim] __attribute__((aligned(32)));
    for (int j = 0; j < dim; j++) {
        center_of_mass[j] = 0.0;
    }

    /* Compute center of mass with SIMD */
    for (int i = 0; i < opt->population_size; i++) {
        const double *restrict pos = opt->population[i].position;
        int j = 0;
        for (; j <= dim - 4; j += 4) {
            __m256d com = _mm256_loadu_pd(&center_of_mass[j]);
            __m256d p = _mm256_loadu_pd(&pos[j]);
            com = _mm256_add_pd(com, p);
            _mm256_storeu_pd(&center_of_mass[j], com);
        }
        for (; j < dim; j++) {
            center_of_mass[j] += pos[j];
        }
    }

    const double inv_pop_size = 1.0 / opt->population_size;
    __m256d inv_pop_vec = _mm256_set1_pd(inv_pop_size);
    int j = 0;
    for (; j <= dim - 4; j += 4) {
        __m256d com = _mm256_loadu_pd(&center_of_mass[j]);
        com = _mm256_mul_pd(com, inv_pop_vec);
        _mm256_storeu_pd(&center_of_mass[j], com);
    }
    for (; j < dim; j++) {
        center_of_mass[j] *= inv_pop_size;
    }

    /* Update positions with SIMD and bounds enforcement */
    const __m256d alpha_vec = _mm256_set1_pd(ALPHA);
    for (int i = 0; i < opt->population_size; i++) {
        double *restrict pos = opt->population[i].position;
        j = 0;
        for (; j <= dim - 4; j += 4) {
            __m256d p = _mm256_loadu_pd(&pos[j]);
            __m256d com = _mm256_loadu_pd(&center_of_mass[j]);
            __m256d direction = _mm256_sub_pd(com, p);
            p = _mm256_fmadd_pd(alpha_vec, direction, p);
            /* Bounds enforcement */
            __m256d min = _mm256_loadu_pd(&opt->bounds[2 * j]);
            __m256d max = _mm256_loadu_pd(&opt->bounds[2 * j + 4]);
            p = _mm256_max_pd(p, min);
            p = _mm256_min_pd(p, max);
            _mm256_storeu_pd(&pos[j], p);
        }
        for (; j < dim; j++) {
            double direction = center_of_mass[j] - pos[j];
            pos[j] += ALPHA * direction;
            pos[j] = pos[j] < opt->bounds[2 * j] ? opt->bounds[2 * j] : 
                     pos[j] > opt->bounds[2 * j + 1] ? opt->bounds[2 * j + 1] : pos[j];
        }
    }
}

/* Update Best Solution with minimal overhead */
void update_best_solution_cfo(Optimizer *restrict opt, double (*objective_function)(const double *)) {
    double best_fitness = opt->best_solution.fitness;
    double *restrict best_pos = opt->best_solution.position;

    for (int i = 0; i < opt->population_size; i++) {
        double new_fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = new_fitness; /* Store fitness for debugging */
        if (new_fitness < best_fitness) {
            best_fitness = new_fitness;
            const double *restrict pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = pos[j];
            }
            opt->best_solution.fitness = best_fitness;
        }
    }
}

/* Main Optimization Function */
void CFO_optimize(Optimizer *restrict opt, double (*objective_function)(const double *)) {
    /* Initialize best solution */
    opt->best_solution.fitness = INFINITY;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = 0.0;
    }

    initialize_population_cfo(opt);
    
    /* Evaluate initial population */
    update_best_solution_cfo(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        central_force_update(opt);
        update_best_solution_cfo(opt, objective_function);
        /* Minimal debug output */
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Value = %.6f\n", iter + 1, opt->best_solution.fitness);
        }
    }
}
