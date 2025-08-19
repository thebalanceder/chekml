#include "PO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.2
#include <stdint.h>

// Xorshift32 RNG for speed
typedef struct {
    uint32_t state;
} Xorshift32;

static inline uint32_t xorshift32_next(Xorshift32 *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static inline double xorshift32_double(Xorshift32 *rng, double min, double max) {
    uint32_t x = xorshift32_next(rng);
    return min + (max - min) * ((double)x / UINT32_MAX);
}

// Election Phase
void election_phase(Optimizer *opt, double *fitness, Winner *winners) {
    int areas = opt->population_size / PARTIES;
    for (int a = 0; a < areas; a++) {
        int start_idx = a;
        int step = areas;
        double min_fitness = INFINITY;
        int min_idx = -1;

        for (int i = start_idx; i < opt->population_size; i += step) {
            if (fitness[i] < min_fitness) {
                min_fitness = fitness[i];
                min_idx = i;
            }
        }

        winners[a].idx = min_idx;
        double *src = opt->population[min_idx].position;
        for (int j = 0; j < opt->dim; j++) {
            winners[a].pos[j] = src[j];
        }
    }
}

// Government Formation Phase
void government_formation_phase(Optimizer *opt, double *fitness, ObjectiveFunction objective_function, double *temp_pos) {
    int areas = opt->population_size / PARTIES;
    Xorshift32 rng = { (uint32_t)(opt->population_size + 1) }; // Simple seed

    __m128d bounds_lo, bounds_hi;
    if (opt->dim == 2) {
        bounds_lo = _mm_set_pd(opt->bounds[2], opt->bounds[0]);
        bounds_hi = _mm_set_pd(opt->bounds[3], opt->bounds[1]);
    }

    for (int p = 0; p < PARTIES; p++) {
        int party_start = p * areas;
        double min_fitness = INFINITY;
        int party_leader_idx = -1;

        for (int a = 0; a < areas; a++) {
            int idx = party_start + a;
            if (fitness[idx] < min_fitness) {
                min_fitness = fitness[idx];
                party_leader_idx = idx;
            }
        }

        double *leader_pos = opt->population[party_leader_idx].position;
        __m128d leader_vec = opt->dim == 2 ? _mm_loadu_pd(leader_pos) : _mm_setzero_pd();

        for (int a = 0; a < areas; a++) {
            int member_idx = party_start + a;
            if (member_idx != party_leader_idx) {
                double r = xorshift32_double(&rng, 0.0, 1.0);
                double *member_pos = opt->population[member_idx].position;

                if (opt->dim == 2) {
                    __m128d member_vec = _mm_loadu_pd(member_pos);
                    __m128d diff = _mm_sub_pd(leader_vec, member_vec);
                    __m128d scaled = _mm_mul_pd(diff, _mm_set1_pd(r));
                    __m128d new_vec = _mm_add_pd(member_vec, scaled);
                    new_vec = _mm_max_pd(new_vec, bounds_lo);
                    new_vec = _mm_min_pd(new_vec, bounds_hi);
                    _mm_storeu_pd(temp_pos, new_vec);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        temp_pos[j] = member_pos[j] + r * (leader_pos[j] - member_pos[j]);
                        temp_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], temp_pos[j]));
                    }
                }

                double new_fitness = objective_function(temp_pos);
                if (new_fitness < fitness[member_idx]) {
                    for (int j = 0; j < opt->dim; j++) {
                        member_pos[j] = temp_pos[j];
                    }
                    fitness[member_idx] = new_fitness;
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Election Campaign Phase
void election_campaign_phase(Optimizer *opt, double *fitness, double *prev_positions, ObjectiveFunction objective_function, double *temp_pos) {
    int areas = opt->population_size / PARTIES;
    Xorshift32 rng = { (uint32_t)(opt->population_size + 2) };

    __m128d bounds_lo, bounds_hi;
    if (opt->dim == 2) {
        bounds_lo = _mm_set_pd(opt->bounds[2], opt->bounds[0]);
        bounds_hi = _mm_set_pd(opt->bounds[3], opt->bounds[1]);
    }

    for (int p = 0; p < PARTIES; p++) {
        int party_start = p * areas;
        for (int a = 0; a < areas; a++) {
            int member_idx = party_start + a;
            double *member_pos = opt->population[member_idx].position;
            double *prev_pos = prev_positions + member_idx * opt->dim;
            double r = xorshift32_double(&rng, 0.0, 1.0);

            if (opt->dim == 2) {
                __m128d member_vec = _mm_loadu_pd(member_pos);
                __m128d prev_vec = _mm_loadu_pd(prev_pos);
                __m128d diff = _mm_sub_pd(member_vec, prev_vec);
                __m128d scaled = _mm_mul_pd(diff, _mm_set1_pd(r));
                __m128d new_vec = _mm_add_pd(member_vec, scaled);
                new_vec = _mm_max_pd(new_vec, bounds_lo);
                new_vec = _mm_min_pd(new_vec, bounds_hi);
                _mm_storeu_pd(temp_pos, new_vec);
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    temp_pos[j] = member_pos[j] + r * (member_pos[j] - prev_pos[j]);
                    temp_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], temp_pos[j]));
                }
            }

            double new_fitness = objective_function(temp_pos);
            if (new_fitness < fitness[member_idx]) {
                for (int j = 0; j < opt->dim; j++) {
                    member_pos[j] = temp_pos[j];
                }
                fitness[member_idx] = new_fitness;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Party Switching Phase
void party_switching_phase(Optimizer *opt, double *fitness, int t) {
    int areas = opt->population_size / PARTIES;
    double psr = (1.0 - t * (1.0 / opt->max_iter)) * LAMBDA_RATE;
    Xorshift32 rng = { (uint32_t)(opt->population_size + t + 3) };

    for (int p = 0; p < PARTIES; p++) {
        for (int a = 0; a < areas; a++) {
            int from_idx = p * areas + a;
            if (xorshift32_double(&rng, 0.0, 1.0) < psr) {
                int to_party = xorshift32_next(&rng) % PARTIES;
                while (to_party == p) {
                    to_party = xorshift32_next(&rng) % PARTIES;
                }
                int to_start = to_party * areas;
                double max_fitness = -INFINITY;
                int to_least_fit_idx = -1;

                for (int i = to_start; i < to_start + areas; i++) {
                    if (fitness[i] > max_fitness) {
                        max_fitness = fitness[i];
                        to_least_fit_idx = i;
                    }
                }

                double *pos1 = opt->population[to_least_fit_idx].position;
                double *pos2 = opt->population[from_idx].position;
                for (int j = 0; j < opt->dim; j++) {
                    double temp = pos1[j];
                    pos1[j] = pos2[j];
                    pos2[j] = temp;
                }
                double temp_fitness = fitness[to_least_fit_idx];
                fitness[to_least_fit_idx] = fitness[from_idx];
                fitness[from_idx] = temp_fitness;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Parliamentarism Phase
void parliamentarism_phase(Optimizer *opt, double *fitness, Winner *winners, ObjectiveFunction objective_function, double *temp_pos) {
    int areas = opt->population_size / PARTIES;
    Xorshift32 rng = { (uint32_t)(opt->population_size + 4) };

    __m128d bounds_lo, bounds_hi;
    if (opt->dim == 2) {
        bounds_lo = _mm_set_pd(opt->bounds[2], opt->bounds[0]);
        bounds_hi = _mm_set_pd(opt->bounds[3], opt->bounds[1]);
    }

    for (int a = 0; a < areas; a++) {
        double *new_winner = temp_pos;
        double *winner_pos = winners[a].pos;
        for (int j = 0; j < opt->dim; j++) {
            new_winner[j] = winner_pos[j];
        }
        int winner_idx = winners[a].idx;

        int to_area = xorshift32_next(&rng) % areas;
        while (to_area == a) {
            to_area = xorshift32_next(&rng) % areas;
        }

        double *to_winner = winners[to_area].pos;
        if (opt->dim == 2) {
            __m128d winner_vec = _mm_load_pd(winner_pos);
            __m128d to_winner_vec = _mm_load_pd(to_winner);
            __m128d distance = _mm_sub_pd(to_winner_vec, winner_vec);
            distance = _mm_andnot_pd(_mm_set1_pd(-0.0), distance); // abs
            double r = xorshift32_double(&rng, -1.0, 1.0);
            __m128d scaled = _mm_mul_pd(distance, _mm_set1_pd(r));
            __m128d new_vec = _mm_add_pd(to_winner_vec, scaled);
            new_vec = _mm_max_pd(new_vec, bounds_lo);
            new_vec = _mm_min_pd(new_vec, bounds_hi);
            _mm_store_pd(new_winner, new_vec);
        } else {
            for (int j = 0; j < opt->dim; j++) {
                double distance = fabs(to_winner[j] - new_winner[j]);
                new_winner[j] = to_winner[j] + (2.0 * xorshift32_double(&rng, 0.0, 1.0) - 1.0) * distance;
                new_winner[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_winner[j]));
            }
        }

        double new_fitness = objective_function(new_winner);
        if (new_fitness < fitness[winner_idx]) {
            double *pop_pos = opt->population[winner_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                pop_pos[j] = new_winner[j];
                winner_pos[j] = new_winner[j];
            }
            fitness[winner_idx] = new_fitness;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void PO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    int areas = opt->population_size / PARTIES;

    // Allocate cache-aligned arrays
    double *fitness = (double *)aligned_alloc(64, opt->population_size * sizeof(double));
    double *aux_fitness = (double *)aligned_alloc(64, opt->population_size * sizeof(double));
    double *prev_positions = (double *)aligned_alloc(64, opt->population_size * opt->dim * sizeof(double));
    Winner *winners = (Winner *)aligned_alloc(64, areas * sizeof(Winner));
    double temp_pos[4] __attribute__((aligned(16))); // Stack allocation for dim <= 4

    // Initialize fitness
    double best_fitness = INFINITY;
    int best_idx = -1;
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = objective_function(opt->population[i].position);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        opt->best_solution.fitness = best_fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = opt->population[best_idx].position[j];
        }
    }
    memcpy(aux_fitness, fitness, opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(prev_positions + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
    }

    // Initial phases
    election_phase(opt, fitness, winners);
    government_formation_phase(opt, fitness, objective_function, temp_pos);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        double *temp_fitness = aux_fitness;
        aux_fitness = fitness;
        fitness = temp_fitness;

        for (int i = 0; i < opt->population_size; i++) {
            memcpy(prev_positions + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        }

        election_campaign_phase(opt, fitness, prev_positions, objective_function, temp_pos);
        party_switching_phase(opt, fitness, t);
        election_phase(opt, fitness, winners);
        government_formation_phase(opt, fitness, objective_function, temp_pos);
        parliamentarism_phase(opt, fitness, winners, objective_function, temp_pos);

        for (int i = 0; i < opt->population_size; i++) {
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    // Free allocated memory
    free(fitness);
    free(aux_fitness);
    free(prev_positions);
    free(winners);
}
