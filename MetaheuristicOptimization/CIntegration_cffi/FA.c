/* FA.c - Optimized Implementation file for Fireworks Algorithm (FWA) Optimization */
#include "FA.h"
#include "generaloptimizer.h"
#include <stdlib.h>    // For malloc(), free()
#include <time.h>      // For time() to seed RNG
#include <math.h>      // For sqrt(), cos(), M_PI
#include <immintrin.h> // For AVX2 intrinsics
#include <omp.h>       // For parallelization
#include <stdint.h>    // For uint64_t

// 🎲 Xorshift RNG implementation
void xorshift_seed(XorshiftState *state, uint64_t seed) {
    state->state = seed ? seed : (uint64_t)time(NULL);
}

static inline uint64_t xorshift_next(XorshiftState *state) {
    uint64_t x = state->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double xorshift_double(XorshiftState *state, double min, double max) {
    return min + (max - min) * ((double)xorshift_next(state) / UINT64_MAX);
}

// Simplified Gaussian approximation (sum of 12 uniforms for near-Gaussian)
double xorshift_gaussian(XorshiftState *state) {
    double sum = 0.0;
    for (int i = 0; i < 12; i++) {
        sum += (double)xorshift_next(state) / UINT64_MAX;
    }
    return sum - 6.0; // Mean 0, variance ~1
}

// Initialize particle positions randomly within bounds (AVX2-optimized)
void initialize_particles(Optimizer *opt, XorshiftState *rng) {
    if (!opt || !opt->population) {
        fprintf(stderr, "🚫 Error: Invalid optimizer or population\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;
    int j;

    #pragma omp parallel private(j)
    {
        XorshiftState thread_rng;
        xorshift_seed(&thread_rng, rng->state ^ (uint64_t)omp_get_thread_num());

        #pragma omp for
        for (int i = 0; i < pop_size; i++) {
            j = 0;
            // Vectorized initialization (AVX2: 4 doubles at a time)
            for (; j <= dim - 4; j += 4) {
                __m256d min_vec = _mm256_loadu_pd(&opt->bounds[2 * j]);
                __m256d max_vec = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                __m256d range = _mm256_sub_pd(max_vec, min_vec);
                __m256d rand = _mm256_set_pd(
                    xorshift_double(&thread_rng, 0.0, 1.0),
                    xorshift_double(&thread_rng, 0.0, 1.0),
                    xorshift_double(&thread_rng, 0.0, 1.0),
                    xorshift_double(&thread_rng, 0.0, 1.0)
                );
                __m256d result = _mm256_fmadd_pd(rand, range, min_vec);
                _mm256_storeu_pd(&opt->population[i].position[j], result);
            }
            // Scalar remainder
            for (; j < dim; j++) {
                double min_bound = opt->bounds[2 * j];
                double max_bound = opt->bounds[2 * j + 1];
                opt->population[i].position[j] = xorshift_double(&thread_rng, min_bound, max_bound);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate fitness for all particles (parallelized)
void evaluate_fitness(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs in evaluate_fitness\n");
        return;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
}

// Generate sparks around the best particle (AVX2-optimized)
void generate_sparks(Optimizer *opt, double *best_particle, double *sparks, XorshiftState *rng) {
    if (!opt || !best_particle || !sparks) {
        fprintf(stderr, "🚫 Error: Invalid inputs in generate_sparks\n");
        return;
    }

    int num_sparks = (int)BETA;
    int dim = opt->dim;
    int j;

    #pragma omp parallel private(j)
    {
        XorshiftState thread_rng;
        xorshift_seed(&thread_rng, rng->state ^ (uint64_t)omp_get_thread_num());

        #pragma omp for
        for (int i = 0; i < num_sparks; i++) {
            j = 0;
            // Vectorized spark generation (AVX2: 4 doubles at a time)
            for (; j <= dim - 4; j += 4) {
                __m256d best_vec = _mm256_loadu_pd(&best_particle[j]);
                __m256d noise = _mm256_set_pd(
                    xorshift_gaussian(&thread_rng),
                    xorshift_gaussian(&thread_rng),
                    xorshift_gaussian(&thread_rng),
                    xorshift_gaussian(&thread_rng)
                );
                __m256d scaled_noise = _mm256_mul_pd(noise, _mm256_set1_pd(ALPHA * DELTA_T));
                __m256d result = _mm256_add_pd(best_vec, scaled_noise);
                _mm256_storeu_pd(&sparks[i * dim + j], result);
            }
            // Scalar remainder
            for (; j < dim; j++) {
                sparks[i * dim + j] = best_particle[j] + ALPHA * xorshift_gaussian(&thread_rng) * DELTA_T;
            }
        }
    }
}

// Comparison function for qsort
typedef struct { double fitness; int index; } ParticleSort;

int compare_particles(const void *a, const void *b) {
    double diff = ((ParticleSort *)a)->fitness - ((ParticleSort *)b)->fitness;
    return (diff > 0) - (diff < 0);
}

// Update particles by combining with sparks and selecting top performers
void update_particles(Optimizer *opt, double (*objective_function)(double *), double *all_positions, double *all_fitness, void *sort_array, XorshiftState *rng) {
    if (!opt || !opt->population || !objective_function || !all_positions || !all_fitness || !sort_array) {
        fprintf(stderr, "🚫 Error: Invalid inputs in update_particles\n");
        return;
    }

    int dim = opt->dim;
    int pop_size = opt->population_size;
    int num_sparks = (int)BETA;
    int total_particles = pop_size + num_sparks;

    // Find best particle
    int min_idx = 0;
    double min_fitness = opt->population[0].fitness;
    for (int i = 1; i < pop_size; i++) {
        if (opt->population[i].fitness < min_fitness) {
            min_fitness = opt->population[i].fitness;
            min_idx = i;
        }
    }

    // Update best solution if improved
    if (min_fitness < opt->best_solution.fitness) {
        opt->best_solution.fitness = min_fitness;
        #pragma omp parallel for
        for (int j = 0; j < dim; j++) {
            opt->best_solution.position[j] = opt->population[min_idx].position[j];
        }
    }

    // Copy particles to all_positions
    #pragma omp parallel for
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) {
            all_positions[i * dim + j] = opt->population[i].position[j];
        }
        all_fitness[i] = opt->population[i].fitness;
    }

    // Generate sparks
    generate_sparks(opt, opt->population[min_idx].position, &all_positions[pop_size * dim], rng);

    // Evaluate spark fitness (parallelized)
    #pragma omp parallel for schedule(dynamic)
    for (int i = pop_size; i < total_particles; i++) {
        all_fitness[i] = objective_function(&all_positions[i * dim]);
    }

    // Apply boundary constraints (AVX2-optimized)
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < total_particles; i++) {
        j = 0;
        for (; j <= dim - 4; j += 4) {
            __m256d pos_vec = _mm256_loadu_pd(&all_positions[i * dim + j]);
            __m256d min_vec = _mm256_loadu_pd(&opt->bounds[2 * j]);
            __m256d max_vec = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
            pos_vec = _mm256_max_pd(_mm256_min_pd(pos_vec, max_vec), min_vec);
            _mm256_storeu_pd(&all_positions[i * dim + j], pos_vec);
        }
        for (; j < dim; j++) {
            double val = all_positions[i * dim + j];
            if (val < opt->bounds[2 * j]) {
                all_positions[i * dim + j] = opt->bounds[2 * j];
            } else if (val > opt->bounds[2 * j + 1]) {
                all_positions[i * dim + j] = opt->bounds[2 * j + 1];
            }
        }
    }

    // Sort particles by fitness
    ParticleSort *sort_array_typed = (ParticleSort *)sort_array;
    #pragma omp parallel for
    for (i = 0; i < total_particles; i++) {
        sort_array_typed[i].fitness = all_fitness[i];
        sort_array_typed[i].index = i;
    }

    qsort(sort_array_typed, total_particles, sizeof(ParticleSort), compare_particles);

    // Select top NUM_PARTICLES
    #pragma omp parallel for
    for (i = 0; i < pop_size; i++) {
        int idx = sort_array_typed[i].index;
        for (j = 0; j < dim; j++) {
            opt->population[i].position[j] = all_positions[idx * dim + j];
        }
        opt->population[i].fitness = all_fitness[idx];
    }
}

// Main Optimization Function
void FA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->bounds || !objective_function) {
        fprintf(stderr, "🚫 Error: Invalid inputs to FA_optimize\n");
        return;
    }

    // Seed random number generator
    XorshiftState rng;
    xorshift_seed(&rng, (uint64_t)time(NULL));

    // Preallocate temporary arrays
    int total_particles = opt->population_size + (int)BETA;
    double *all_positions = (double *)malloc(total_particles * opt->dim * sizeof(double));
    double *all_fitness = (double *)malloc(total_particles * sizeof(double));
    ParticleSort *sort_array = (ParticleSort *)malloc(total_particles * sizeof(ParticleSort));
    if (!all_positions || !all_fitness || !sort_array) {
        fprintf(stderr, "🚫 Error: Memory allocation failed in FA_optimize\n");
        free(all_positions);
        free(all_fitness);
        free(sort_array);
        return;
    }

    // Initialize particles
    initialize_particles(opt, &rng);

    // Evaluate initial fitness
    evaluate_fitness(opt, objective_function);

    // Set initial best solution
    int min_idx = 0;
    double min_fitness = opt->population[0].fitness;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->population[i].fitness < min_fitness) {
            min_fitness = opt->population[i].fitness;
            min_idx = i;
        }
    }
    opt->best_solution.fitness = min_fitness;
    #pragma omp parallel for
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[min_idx].position[j];
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        update_particles(opt, objective_function, all_positions, all_fitness, sort_array, &rng);
        printf("🔄 Iteration %d: Best Fitness = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Clean up
    free(all_positions);
    free(all_fitness);
    free(sort_array);
}
