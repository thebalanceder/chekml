#include "GWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2 for SIMD

// Xorshift128+ RNG state
typedef struct {
    uint64_t s[2];
} XorshiftState;

// Initialize Xorshift128+ state
static inline void xorshift_init(XorshiftState *state, uint64_t seed) {
    state->s[0] = seed ^ 0x123456789ABCDEFULL;
    state->s[1] = seed ^ 0xFEDCBA9876543210ULL;
}

// Xorshift128+ random number generator
static inline uint64_t xorshift_next(XorshiftState *state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;
    state->s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return state->s[1] + s0;
}

// Fast random double in [0, 1)
static inline double gwo_rand_double(void *rng_state) {
    XorshiftState *state = (XorshiftState *)rng_state;
    return (double)(xorshift_next(state) >> 11) / (double)(1ULL << 53);
}

// Initialize the population of search agents
void gwo_initialize_population(Optimizer *opt, void *rng_state) {
    int pop_size = opt->population_size;
    int dim = opt->dim;
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = lb + gwo_rand_double(rng_state) * (ub - lb);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Update Alpha, Beta, and Delta wolves based on fitness
void update_hierarchy(Optimizer *opt, Solution *beta_solution, Solution *delta_solution) {
    double alpha_score = opt->best_solution.fitness;
    double beta_score = beta_solution->fitness;
    double delta_score = delta_solution->fitness;
    int alpha_idx = -1, beta_idx = -1, delta_idx = -1;
    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Find Alpha, Beta, Delta
    for (int i = 0; i < pop_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < alpha_score) {
            delta_score = beta_score;
            delta_idx = beta_idx;
            beta_score = alpha_score;
            beta_idx = alpha_idx;
            alpha_score = fitness;
            alpha_idx = i;
        } else if (fitness < beta_score && fitness > alpha_score) {
            delta_score = beta_score;
            delta_idx = beta_idx;
            beta_score = fitness;
            beta_idx = i;
        } else if (fitness < delta_score && fitness > beta_score) {
            delta_score = fitness;
            delta_idx = i;
        }
    }

    // Update Alpha (best_solution)
    if (alpha_idx >= 0) {
        opt->best_solution.fitness = alpha_score;
        double *alpha_pos = opt->best_solution.position;
        double *src_pos = opt->population[alpha_idx].position;
        for (int j = 0; j < dim; j++) {
            alpha_pos[j] = src_pos[j];
        }
    }

    // Update Beta
    if (beta_idx >= 0) {
        beta_solution->fitness = beta_score;
        double *beta_pos = beta_solution->position;
        double *src_pos = opt->population[beta_idx].position;
        for (int j = 0; j < dim; j++) {
            beta_pos[j] = src_pos[j];
        }
    }

    // Update Delta
    if (delta_idx >= 0) {
        delta_solution->fitness = delta_score;
        double *delta_pos = delta_solution->position;
        double *src_pos = opt->population[delta_idx].position;
        for (int j = 0; j < dim; j++) {
            delta_pos[j] = src_pos[j];
        }
    }
}

// Update positions using SIMD (SSE2) for dimensions divisible by 2
void gwo_update_positions(Optimizer *opt, double a, Solution *beta_solution, Solution *delta_solution) {
    double *alpha_pos = opt->best_solution.position;
    double *beta_pos = beta_solution->position;
    double *delta_pos = delta_solution->position;
    int dim = opt->dim;
    int pop_size = opt->population_size;
    XorshiftState *rng_state = (XorshiftState *)opt->optimize; // Reuse optimize pointer for RNG state

    // Precompute random numbers for the iteration
    double *rand_buffer = (double *)malloc(6 * pop_size * dim * sizeof(double));
    for (int k = 0; k < 6 * pop_size * dim; k++) {
        rand_buffer[k] = gwo_rand_double(rng_state);
    }
    int rand_idx = 0;

    // SIMD processing for pairs of dimensions
    int dim_pairs = dim / 2;
    for (int i = 0; i < pop_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < dim_pairs * 2; j += 2) {
            // Load two dimensions
            __m128d pos_vec = _mm_loadu_pd(&pos[j]);
            __m128d alpha_vec = _mm_loadu_pd(&alpha_pos[j]);
            __m128d beta_vec = _mm_loadu_pd(&beta_pos[j]);
            __m128d delta_vec = _mm_loadu_pd(&delta_pos[j]);

            // Alpha update
            double r1 = rand_buffer[rand_idx++];
            double r2 = rand_buffer[rand_idx++];
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            __m128d C1_vec = _mm_set1_pd(C1);
            __m128d D_alpha = _mm_sub_pd(_mm_mul_pd(C1_vec, alpha_vec), pos_vec);
            D_alpha = _mm_max_pd(D_alpha, _mm_sub_pd(_mm_setzero_pd(), D_alpha)); // fabs
            __m128d X1 = _mm_sub_pd(alpha_vec, _mm_mul_pd(_mm_set1_pd(A1), D_alpha));

            // Beta update
            r1 = rand_buffer[rand_idx++];
            r2 = rand_buffer[rand_idx++];
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            __m128d C2_vec = _mm_set1_pd(C2);
            __m128d D_beta = _mm_sub_pd(_mm_mul_pd(C2_vec, beta_vec), pos_vec);
            D_beta = _mm_max_pd(D_beta, _mm_sub_pd(_mm_setzero_pd(), D_beta));
            __m128d X2 = _mm_sub_pd(beta_vec, _mm_mul_pd(_mm_set1_pd(A2), D_beta));

            // Delta update
            r1 = rand_buffer[rand_idx++];
            r2 = rand_buffer[rand_idx++];
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            __m128d C3_vec = _mm_set1_pd(C3);
            __m128d D_delta = _mm_sub_pd(_mm_mul_pd(C3_vec, delta_vec), pos_vec);
            D_delta = _mm_max_pd(D_delta, _mm_sub_pd(_mm_setzero_pd(), D_delta));
            __m128d X3 = _mm_sub_pd(delta_vec, _mm_mul_pd(_mm_set1_pd(A3), D_delta));

            // Combine and store
            __m128d sum = _mm_add_pd(X1, _mm_add_pd(X2, X3));
            __m128d result = _mm_div_pd(sum, _mm_set1_pd(3.0));
            _mm_storeu_pd(&pos[j], result);
        }

        // Handle remaining dimension (if dim is odd)
        for (int j = dim_pairs * 2; j < dim; j++) {
            double r1 = rand_buffer[rand_idx++];
            double r2 = rand_buffer[rand_idx++];
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            double D_alpha = fabs(C1 * alpha_pos[j] - pos[j]);
            double X1 = alpha_pos[j] - A1 * D_alpha;

            r1 = rand_buffer[rand_idx++];
            r2 = rand_buffer[rand_idx++];
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            double D_beta = fabs(C2 * beta_pos[j] - pos[j]);
            double X2 = beta_pos[j] - A2 * D_beta;

            r1 = rand_buffer[rand_idx++];
            r2 = rand_buffer[rand_idx++];
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            double D_delta = fabs(C3 * delta_pos[j] - pos[j]);
            double X3 = delta_pos[j] - A3 * D_delta;

            pos[j] = (X1 + X2 + X3) / 3.0;
        }
    }
    free(rand_buffer);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize Xorshift128+ RNG
    XorshiftState rng_state;
    xorshift_init(&rng_state, (uint64_t)time(NULL));

    // Allocate Beta and Delta solutions
    Solution beta_solution;
    Solution delta_solution;
    beta_solution.position = (double *)malloc(opt->dim * sizeof(double));
    delta_solution.position = (double *)malloc(opt->dim * sizeof(double));
    beta_solution.fitness = INFINITY;
    delta_solution.fitness = INFINITY;

    // Initialize population
    gwo_initialize_population(opt, &rng_state);

    // Precompute a values
    double *a_values = (double *)malloc(opt->max_iter * sizeof(double));
    for (int iter = 0; iter < opt->max_iter; iter++) {
        a_values[iter] = GWO_A_MAX - ((double)iter / opt->max_iter) * (GWO_A_MAX - GWO_A_MIN);
    }

    // Temporarily store RNG state in optimize pointer
    void *original_optimize = opt->optimize;
    opt->optimize = (void *)&rng_state;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness (consider OpenMP for parallelization)
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }

        // Update hierarchy
        update_hierarchy(opt, &beta_solution, &delta_solution);

        // Update positions
        gwo_update_positions(opt, a_values[iter], &beta_solution, &delta_solution);

        // Log progress
        printf("Iteration %d: Best Score = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Restore original optimize pointer
    opt->optimize = original_optimize;

    // Clean up
    free(a_values);
    free(beta_solution.position);
    free(delta_solution.position);
}
