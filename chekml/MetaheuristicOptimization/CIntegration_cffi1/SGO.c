/* SGO.c - Extreme Performance Implementation for Squid Game Optimization */
#include "SGO.h"
#include "generaloptimizer.h"
#include <immintrin.h>  // AVX2 intrinsics
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Compiler-specific optimizations
#if defined(__GNUC__) || defined(__clang__)
#define INLINE __attribute__((always_inline)) inline
#define RESTRICT __restrict__
#else
#define INLINE inline
#define RESTRICT
#endif

// Align memory for AVX2 (32-byte alignment)
#define ALIGN 32

// Initialize random buffer
static void init_rand_buffer(double *rand_buffer, int size) {
    for (int i = 0; i < size; i++) {
        rand_buffer[i] = (double)rand() / RAND_MAX;
    }
}

// Initialize Players with SIMD
void sgo_initialize_players(Optimizer *opt, double *rand_buffer, int *rand_idx) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    __m256d *bounds = (__m256d *)opt->bounds; // Assumes bounds is 32-byte aligned
    int ri = *rand_idx;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j += 4) {
            if (j + 4 <= dim) {
                __m256d r = _mm256_loadu_pd(&rand_buffer[ri]);
                ri = (ri + 4) % RAND_BUFFER_SIZE;
                __m256d lower = _mm256_broadcast_sd(&opt->bounds[2 * j]);
                __m256d upper = _mm256_broadcast_sd(&opt->bounds[2 * j + 1]);
                __m256d range = _mm256_sub_pd(upper, lower);
                __m256d scaled = _mm256_mul_pd(r, range);
                __m256d result = _mm256_add_pd(lower, scaled);
                _mm256_storeu_pd(&pos[j], result);
            } else {
                for (int k = j; k < dim; k++) {
                    pos[k] = opt->bounds[2 * k] + rand_buffer[ri] * (opt->bounds[2 * k + 1] - opt->bounds[2 * k]);
                    ri = (ri + 1) % RAND_BUFFER_SIZE;
                }
            }
        }
        opt->population[i].fitness = INFINITY;
    }
    *rand_idx = ri;
    enforce_bound_constraints(opt);
}

// Divide Teams with optimized shuffle
void sgo_divide_teams(Optimizer *opt, int *RESTRICT offensive_indices, int *RESTRICT defensive_indices, int *offensive_size) {
    *offensive_size = (int)(opt->population_size * ATTACK_RATE);
    int *indices = offensive_indices;
    const int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        indices[i] = i;
    }

    // Fisher-Yates shuffle
    for (int i = pop_size - 1; i > 0; i--) {
        int j = (int)(rand() % (i + 1));
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = *offensive_size; i < pop_size; i++) {
        defensive_indices[i - *offensive_size] = indices[i];
    }
}

// Simulate Fights with SIMD
void sgo_simulate_fights(Optimizer *opt, int *RESTRICT offensive_indices, int *RESTRICT defensive_indices, 
                         int size, double *rand_buffer, int *rand_idx) {
    const int dim = opt->dim;
    int ri = *rand_idx;
    double *temp_offensive __attribute__((aligned(ALIGN)));
    double *temp_defensive __attribute__((aligned(ALIGN)));
    int ret;

    ret = posix_memalign((void **)&temp_offensive, ALIGN, dim * sizeof(double));
    if (ret != 0) {
        fprintf(stderr, "Memory allocation failed for temp_offensive\n");
        exit(1);
    }
    ret = posix_memalign((void **)&temp_defensive, ALIGN, dim * sizeof(double));
    if (ret != 0) {
        fprintf(stderr, "Memory allocation failed for temp_defensive\n");
        free(temp_offensive);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        double *off_pos = opt->population[offensive_indices[i]].position;
        double *def_pos = opt->population[defensive_indices[i]].position;
        __m256d r1 = _mm256_set1_pd(rand_buffer[ri]);
        ri = (ri + 1) % RAND_BUFFER_SIZE;

        for (int j = 0; j < dim; j += 4) {
            if (j + 4 <= dim) {
                __m256d off = _mm256_loadu_pd(&off_pos[j]);
                __m256d def = _mm256_loadu_pd(&def_pos[j]);
                __m256d diff = _mm256_sub_pd(off, def);
                __m256d move = _mm256_mul_pd(_mm256_set1_pd(FIGHT_INTENSITY), _mm256_mul_pd(diff, r1));
                __m256d resist = _mm256_mul_pd(_mm256_set1_pd(DEFENSE_STRENGTH), 
                                              _mm256_mul_pd(diff, _mm256_sub_pd(_mm256_set1_pd(1.0), r1)));
                __m256d new_off = _mm256_add_pd(off, move);
                __m256d new_def = _mm256_add_pd(def, resist);

                // Bounds checking
                __m256d lower = _mm256_loadu_pd(&opt->bounds[2 * j]);
                __m256d upper = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                new_off = _mm256_max_pd(new_off, lower);
                new_off = _mm256_min_pd(new_off, upper);
                new_def = _mm256_max_pd(new_def, lower);
                new_def = _mm256_min_pd(new_def, upper);

                _mm256_storeu_pd(&temp_offensive[j], new_off);
                _mm256_storeu_pd(&temp_defensive[j], new_def);
            } else {
                for (int k = j; k < dim; k++) {
                    double diff = off_pos[k] - def_pos[k];
                    temp_offensive[k] = off_pos[k] + FIGHT_INTENSITY * diff * rand_buffer[ri];
                    temp_defensive[k] = def_pos[k] + DEFENSE_STRENGTH * diff * (1.0 - rand_buffer[ri]);
                    ri = (ri + 1) % RAND_BUFFER_SIZE;
                    temp_offensive[k] = fmax(opt->bounds[2 * k], fmin(opt->bounds[2 * k + 1], temp_offensive[k]));
                    temp_defensive[k] = fmax(opt->bounds[2 * k], fmin(opt->bounds[2 * k + 1], temp_defensive[k]));
                }
            }
        }

        memcpy(off_pos, temp_offensive, dim * sizeof(double));
        memcpy(def_pos, temp_defensive, dim * sizeof(double));
    }

    free(temp_offensive);
    free(temp_defensive);
    *rand_idx = ri;
}

// Determine Winners
void sgo_determine_winners(Optimizer *opt, int *RESTRICT offensive_indices, int offensive_size, 
                           int *RESTRICT defensive_indices, int defensive_size, int *RESTRICT winners, int *winner_count) {
    *winner_count = 0;
    int min_size = (offensive_size < defensive_size) ? offensive_size : defensive_size;

    for (int i = 0; i < min_size; i++) {
        int off_idx = offensive_indices[i];
        int def_idx = defensive_indices[i];
        double off_fitness = opt->population[off_idx].fitness;
        double def_fitness = opt->population[def_idx].fitness;

        if (off_fitness < def_fitness * WIN_THRESHOLD) {
            winners[(*winner_count)++] = off_idx;
        } else if (def_fitness < off_fitness * WIN_THRESHOLD) {
            winners[(*winner_count)++] = def_idx;
        }
    }
}

// Update Positions with SIMD
void sgo_update_positions(Optimizer *opt, int *RESTRICT winners, int winner_count, double *rand_buffer, int *rand_idx) {
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    int ri = *rand_idx;

    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        int is_winner = 0;
        for (int w = 0; w < winner_count; w++) {
            if (winners[w] == i) {
                is_winner = 1;
                break;
            }
        }

        if (is_winner && opt->best_solution.fitness != INFINITY) {
            __m256d r2 = _mm256_set1_pd(rand_buffer[ri]);
            ri = (ri + 1) % RAND_BUFFER_SIZE;
            for (int j = 0; j < dim; j += 4) {
                if (j + 4 <= dim) {
                    __m256d p = _mm256_loadu_pd(&pos[j]);
                    __m256d best = _mm256_loadu_pd(&opt->best_solution.position[j]);
                    __m256d diff = _mm256_sub_pd(best, p);
                    __m256d update = _mm256_mul_pd(_mm256_set1_pd(ATTACK_RATE), _mm256_mul_pd(r2, diff));
                    __m256d new_pos = _mm256_add_pd(p, update);
                    __m256d lower = _mm256_loadu_pd(&opt->bounds[2 * j]);
                    __m256d upper = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                    new_pos = _mm256_max_pd(new_pos, lower);
                    new_pos = _mm256_min_pd(new_pos, upper);
                    _mm256_storeu_pd(&pos[j], new_pos);
                } else {
                    for (int k = j; k < dim; k++) {
                        pos[k] += ATTACK_RATE * rand_buffer[ri] * (opt->best_solution.position[k] - pos[k]);
                        pos[k] = fmax(opt->bounds[2 * k], fmin(opt->bounds[2 * k + 1], pos[k]));
                        ri = (ri + 1) % RAND_BUFFER_SIZE;
                    }
                }
            }
        } else {
            for (int j = 0; j < dim; j += 4) {
                if (j + 4 <= dim) {
                    __m256d r = _mm256_loadu_pd(&rand_buffer[ri]);
                    ri = (ri + 4) % RAND_BUFFER_SIZE;
                    __m256d range = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                    range = _mm256_sub_pd(range, _mm256_loadu_pd(&opt->bounds[2 * j]));
                    __m256d update = _mm256_mul_pd(_mm256_set1_pd(FIGHT_INTENSITY), _mm256_mul_pd(r, range));
                    __m256d new_pos = _mm256_add_pd(_mm256_loadu_pd(&pos[j]), update);
                    __m256d lower = _mm256_loadu_pd(&opt->bounds[2 * j]);
                    __m256d upper = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
                    new_pos = _mm256_max_pd(new_pos, lower);
                    new_pos = _mm256_min_pd(new_pos, upper);
                    _mm256_storeu_pd(&pos[j], new_pos);
                } else {
                    for (int k = j; k < dim; k++) {
                        pos[k] += FIGHT_INTENSITY * rand_buffer[ri] * (opt->bounds[2 * k + 1] - opt->bounds[2 * k]);
                        pos[k] = fmax(opt->bounds[2 * k], fmin(opt->bounds[2 * k + 1], pos[k]));
                        ri = (ri + 1) % RAND_BUFFER_SIZE;
                    }
                }
            }
        }
    }
    *rand_idx = ri;
}

// Main Optimization Function
void SGO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *rand_buffer __attribute__((aligned(ALIGN)));
    int ret = posix_memalign((void **)&rand_buffer, ALIGN, RAND_BUFFER_SIZE * sizeof(double));
    if (ret != 0) {
        fprintf(stderr, "Memory allocation failed for rand_buffer\n");
        exit(1);
    }
    init_rand_buffer(rand_buffer, RAND_BUFFER_SIZE);
    int rand_idx = 0;

    int offensive_indices[opt->population_size];
    int defensive_indices[opt->population_size];
    int winners[opt->population_size];
    int offensive_size, winner_count;

    sgo_initialize_players(opt, rand_buffer, &rand_idx);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        double min_fitness = INFINITY;
        int min_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }

        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
        }

        sgo_divide_teams(opt, offensive_indices, defensive_indices, &offensive_size);
        int fight_count = (offensive_size < (opt->population_size - offensive_size)) 
                         ? offensive_size : (opt->population_size - offensive_size);
        sgo_simulate_fights(opt, offensive_indices, defensive_indices, fight_count, rand_buffer, &rand_idx);
        sgo_determine_winners(opt, offensive_indices, offensive_size, 
                             defensive_indices, opt->population_size - offensive_size, 
                             winners, &winner_count);
        sgo_update_positions(opt, winners, winner_count, rand_buffer, &rand_idx);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(rand_buffer);
}
