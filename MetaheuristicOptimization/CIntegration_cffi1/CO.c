#include "CO.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <string.h>
#include <stdio.h>

// 🔧 Fixed-size data structures for stack allocation
typedef struct {
    double positions[MAX_POP * MAX_DIM];  // SoA: flattened positions
    double fitness[MAX_POP];             // Fitness values
    double best_position[MAX_DIM];       // Best solution
    double best_fitness;                 // Best fitness
    double bounds[MAX_DIM * 2];          // Bounds: [min0, max0, min1, max1, ...]
    double bound_ranges[MAX_DIM];        // Cached: max - min per dimension
    int32_t population_size;             // Current population size
    int32_t dim;                         // Dimensions
    int32_t max_iter;                    // Max iterations
} FastOptimizer;

// 🌊 Xorshift RNG for speed
static uint64_t rng_state = 123456789;
static inline double fast_rand_co(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 12) / (1ULL << 52);  // [0, 1)
}

// 🐦 Initialize Cuckoo Population
static inline void initialize_cuckoos(FastOptimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        int base = i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            opt->positions[base + j] = opt->bounds[2 * j] + fast_rand_co() * opt->bound_ranges[j];
        }
        opt->fitness[i] = INFINITY;
    }
    opt->best_fitness = INFINITY;
}

// 🥚 Lay Eggs for Each Cuckoo
static inline void lay_eggs(FastOptimizer *opt, double *egg_positions, int32_t *num_eggs, int32_t *total_eggs) {
    *total_eggs = 0;
    for (int i = 0; i < opt->population_size; i++) {
        num_eggs[i] = MIN_EGGS + (int)(fast_rand_co() * (MAX_EGGS - MIN_EGGS + 1));
        *total_eggs += num_eggs[i];
    }

    int egg_idx = 0;
    for (int i = 0; i < opt->population_size; i++) {
        double radius_factor = ((double)num_eggs[i] / *total_eggs) * RADIUS_COEFF;
        int base = i * opt->dim;
        for (int k = 0; k < num_eggs[i]; k++) {
            double scalar = fast_rand_co();
            double angle = k * (2 * PI / num_eggs[i]);
            double cos_a = cos(angle), sin_a = sin(angle);
            int sign = (fast_rand_co() < 0.5) ? 1 : -1;

            #pragma omp simd
            for (int j = 0; j < opt->dim; j++) {
                double radius = radius_factor * scalar * opt->bound_ranges[j];
                double add = sign * radius * cos_a + radius * sin_a;
                double new_pos = opt->positions[base + j] + add;
                new_pos = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos));
                egg_positions[egg_idx * opt->dim + j] = new_pos;
            }
            egg_idx++;
        }
    }
}

// 🏆 Select Best Cuckoos (Min-Heap)
static inline void select_best_cuckoos(FastOptimizer *opt, double *positions, double *fitness, int num_positions) {
    if (num_positions <= MAX_CUCKOOS) {
        memcpy(opt->positions, positions, num_positions * opt->dim * sizeof(double));
        memcpy(opt->fitness, fitness, num_positions * sizeof(double));
        opt->population_size = num_positions;
        return;
    }

    typedef struct { double fitness; int index; } HeapNode;
    HeapNode heap[MAX_CUCKOOS];
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        heap[i].fitness = fitness[i];
        heap[i].index = i;
    }

    // Build min-heap
    for (int i = MAX_CUCKOOS / 2 - 1; i >= 0; i--) {
        int k = i;
        while (k * 2 + 1 < MAX_CUCKOOS) {
            int child = k * 2 + 1;
            if (child + 1 < MAX_CUCKOOS && heap[child + 1].fitness < heap[child].fitness) child++;
            if (heap[k].fitness <= heap[child].fitness) break;
            HeapNode temp = heap[k];
            heap[k] = heap[child];
            heap[child] = temp;
            k = child;
        }
    }

    // Process remaining elements
    for (int i = MAX_CUCKOOS; i < num_positions; i++) {
        if (fitness[i] < heap[0].fitness) {
            heap[0].fitness = fitness[i];
            heap[0].index = i;
            int k = 0;
            while (k * 2 + 1 < MAX_CUCKOOS) {
                int child = k * 2 + 1;
                if (child + 1 < MAX_CUCKOOS && heap[child + 1].fitness < heap[child].fitness) child++;
                if (heap[k].fitness <= heap[child].fitness) break;
                HeapNode temp = heap[k];
                heap[k] = heap[child];
                heap[child] = temp;
                k = child;
            }
        }
    }

    // Copy top MAX_CUCKOOS
    for (int i = 0; i < MAX_CUCKOOS; i++) {
        int idx = heap[i].index;
        memcpy(&opt->positions[i * opt->dim], &positions[idx * opt->dim], opt->dim * sizeof(double));
        opt->fitness[i] = fitness[idx];
    }
    opt->population_size = MAX_CUCKOOS;
}

// 🌐 Simplified Convergence Check and Migration
static inline int cluster_and_migrate(FastOptimizer *opt) {
    // Find best cuckoo
    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (opt->fitness[i] < opt->fitness[best_idx]) best_idx = i;
    }

    // Check convergence (max distance to best)
    double max_dist = 0.0;
    int best_base = best_idx * opt->dim;
    for (int i = 0; i < opt->population_size; i++) {
        double dist = 0.0;
        int base = i * opt->dim;
        #pragma omp simd reduction(+:dist)
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->positions[base + j] - opt->positions[best_base + j];
            dist += diff * diff;
        }
        max_dist = fmax(max_dist, dist);
    }
    if (max_dist < VARIANCE_THRESHOLD) return 1;

    // Migrate toward best (AVX for dim >= 4)
    if (opt->dim >= 4) {
        for (int i = 0; i < opt->population_size; i++) {
            int base = i * opt->dim;
            for (int j = 0; j <= opt->dim - 4; j += 4) {
                __m256d pos = _mm256_loadu_pd(&opt->positions[base + j]);
                __m256d best = _mm256_loadu_pd(&opt->positions[best_base + j]);
                __m256d delta = _mm256_sub_pd(best, pos);
                __m256d coeff = _mm256_set1_pd(MOTION_COEFF * fast_rand_co());
                __m256d update = _mm256_mul_pd(coeff, delta);
                pos = _mm256_add_pd(pos, update);
                // Bounds clamping
                __m256d min_bounds = _mm256_loadu_pd(&opt->bounds[2 * j]);
                __m256d max_bounds = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                pos = _mm256_max_pd(min_bounds, _mm256_min_pd(max_bounds, pos));
                _mm256_storeu_pd(&opt->positions[base + j], pos);
            }
            for (int j = (opt->dim / 4) * 4; j < opt->dim; j++) {
                double delta = opt->positions[best_base + j] - opt->positions[base + j];
                opt->positions[base + j] += MOTION_COEFF * fast_rand_co() * delta;
                opt->positions[base + j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], opt->positions[base + j]));
            }
        }
    } else {
        for (int i = 0; i < opt->population_size; i++) {
            int base = i * opt->dim;
            #pragma omp simd
            for (int j = 0; j < opt->dim; j++) {
                double delta = opt->positions[best_base + j] - opt->positions[base + j];
                opt->positions[base + j] += MOTION_COEFF * fast_rand_co() * delta;
                opt->positions[base + j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], opt->positions[base + j]));
            }
        }
    }
    return 0;
}

// 🚀 Main Optimization Function
__attribute__((hot))
void CO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Convert to FastOptimizer
    FastOptimizer fopt;
    fopt.population_size = opt->population_size;
    fopt.dim = opt->dim;
    fopt.max_iter = opt->max_iter;
    fopt.best_fitness = INFINITY;
    if (fopt.population_size > MAX_POP || fopt.dim > MAX_DIM) {
        fprintf(stderr, "Error: population_size (%d) or dim (%d) exceeds max (%d, %d)\n",
                fopt.population_size, fopt.dim, MAX_POP, MAX_DIM);
        return;
    }

    // Copy bounds and cache ranges
    memcpy(fopt.bounds, opt->bounds, 2 * fopt.dim * sizeof(double));
    for (int j = 0; j < fopt.dim; j++) {
        fopt.bound_ranges[j] = fopt.bounds[2 * j + 1] - fopt.bounds[2 * j];
    }

    // Copy initial population
    for (int i = 0; i < fopt.population_size; i++) {
        memcpy(&fopt.positions[i * fopt.dim], opt->population[i].position, fopt.dim * sizeof(double));
    }

    // Stack-allocated buffers
    double egg_positions[MAX_TOTAL_EGGS * MAX_DIM];
    double egg_fitness[MAX_TOTAL_EGGS];
    double all_positions[(MAX_POP + MAX_TOTAL_EGGS) * MAX_DIM];
    double all_fitness[MAX_POP + MAX_TOTAL_EGGS];
    int32_t num_eggs[MAX_POP];

    // Initialize
    initialize_cuckoos(&fopt);

    // Evaluate initial population
    for (int i = 0; i < fopt.population_size; i++) {
        fopt.fitness[i] = objective_function(&fopt.positions[i * fopt.dim]);
        if (fopt.fitness[i] < fopt.best_fitness) {
            fopt.best_fitness = fopt.fitness[i];
            memcpy(fopt.best_position, &fopt.positions[i * fopt.dim], fopt.dim * sizeof(double));
        }
    }

    // Main loop
    for (int iter = 0; iter < fopt.max_iter; iter++) {
        // Lay eggs
        int32_t total_eggs = 0;
        lay_eggs(&fopt, egg_positions, num_eggs, &total_eggs);

        // Evaluate eggs
        #pragma omp parallel for if(total_eggs > 100)
        for (int i = 0; i < total_eggs; i++) {
            egg_fitness[i] = objective_function(&egg_positions[i * fopt.dim]);
        }

        // Combine cuckoos and eggs
        int total_positions = fopt.population_size + total_eggs;
        memcpy(all_positions, fopt.positions, fopt.population_size * fopt.dim * sizeof(double));
        memcpy(&all_positions[fopt.population_size * fopt.dim], egg_positions, total_eggs * fopt.dim * sizeof(double));
        memcpy(all_fitness, fopt.fitness, fopt.population_size * sizeof(double));
        memcpy(&all_fitness[fopt.population_size], egg_fitness, total_eggs * sizeof(double));

        // Select best cuckoos
        select_best_cuckoos(&fopt, all_positions, all_fitness, total_positions);

        // Cluster and migrate
        int stop = cluster_and_migrate(&fopt);

        // Update best solution
        int worst_idx = 0, second_worst_idx = (fopt.population_size > 1) ? 1 : 0;
        for (int i = 0; i < fopt.population_size; i++) {
            fopt.fitness[i] = objective_function(&fopt.positions[i * fopt.dim]);
            if (fopt.fitness[i] < fopt.best_fitness) {
                fopt.best_fitness = fopt.fitness[i];
                memcpy(fopt.best_position, &fopt.positions[i * fopt.dim], fopt.dim * sizeof(double));
            }
            if (fopt.fitness[i] > fopt.fitness[worst_idx]) {
                second_worst_idx = worst_idx;
                worst_idx = i;
            } else if (i != worst_idx && fopt.fitness[i] > fopt.fitness[second_worst_idx]) {
                second_worst_idx = i;
            }
        }

        // Replace worst with best
        if (fopt.fitness[worst_idx] > fopt.best_fitness) {
            memcpy(&fopt.positions[worst_idx * fopt.dim], fopt.best_position, fopt.dim * sizeof(double));
            fopt.fitness[worst_idx] = fopt.best_fitness;
        }

        // Replace second-worst with randomized best
        if (fopt.population_size > 1) {
            int base = second_worst_idx * fopt.dim;
            #pragma omp simd
            for (int j = 0; j < fopt.dim; j++) {
                double new_pos = fopt.best_position[j] * fast_rand_co();
                new_pos = fmax(fopt.bounds[2 * j], fmin(fopt.bounds[2 * j + 1], new_pos));
                fopt.positions[base + j] = new_pos;
            }
            fopt.fitness[second_worst_idx] = objective_function(&fopt.positions[base]);
        }

        if (stop) break;

        printf("Iteration %d: Best Value = %f\n", iter + 1, fopt.best_fitness);
    }

    // Copy results back to Optimizer
    opt->best_solution.fitness = fopt.best_fitness;
    memcpy(opt->best_solution.position, fopt.best_position, fopt.dim * sizeof(double));
    for (int i = 0; i < fopt.population_size; i++) {
        memcpy(opt->population[i].position, &fopt.positions[i * fopt.dim], fopt.dim * sizeof(double));
        opt->population[i].fitness = fopt.fitness[i];
    }
    opt->population_size = fopt.population_size;
}
