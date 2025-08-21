#include "EPO.h"
#include "generaloptimizer.h"
#include <immintrin.h> // For SSE4.2 intrinsics
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Lookup table for chaotic strategy (logistic map for t % 10)
static const double chaotic_lookup[10] = {
    0.7,          // x = 0.7
    0.84,         // x = 4 * 0.7 * (1 - 0.7)
    0.5376,       // x = 4 * 0.84 * (1 - 0.84)
    0.99526656,   // x = 4 * 0.5376 * (1 - 0.5376)
    0.018839808,  // etc.
    0.07388576,
    0.273297305,
    0.795147087,
    0.65097507,
    0.908848309
};

// Initialize penguin positions randomly
void initialize_penguins(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute huddle boundary and temperature profile
static inline void compute_huddle_boundary(double *T_prime, double *R, int t, int max_iter) {
    *T_prime = 2.0 - ((double)t / max_iter);
    *R = rand_double(0.0, 1.0);
}

// Adapt control parameters (f, l, m_param) based on selected strategy
static inline void adapt_parameters(double *f, double *l, double *m_param, int strategy, double t_norm, int t) {
    if (strategy == STRATEGY_LINEAR) {
        *f = 2.0 - t_norm * 1.5;
        *l = 1.5 - t_norm * 1.0;
        *m_param = 0.5 + t_norm * 0.3;
    } else if (strategy == STRATEGY_EXPONENTIAL) {
        *f = 2.0 * exp(-t_norm * 2.0);
        *l = 1.5 * exp(-t_norm * 3.0);
        *m_param = 0.5 * (1.0 + tanh(t_norm * 4.0));
    } else { // STRATEGY_CHAOTIC
        double x = chaotic_lookup[t % 10];
        *f = 1.5 + x * 0.5;
        *l = 1.0 + x * 0.5;
        *m_param = 0.3 + x * 0.4;
    }
}

// Update strategy selection probabilities based on historical success
static inline void update_strategy_probabilities(double *strategy_probs, double *strategy_success) {
    double total_success = strategy_success[0] + strategy_success[1] + strategy_success[2] + 1e-10;
    double inv_total = 1.0 / total_success;
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] = strategy_success[i] * inv_total;
        strategy_probs[i] = strategy_probs[i] < 0.1 ? 0.1 : (strategy_probs[i] > 0.9 ? 0.9 : strategy_probs[i]);
    }
    double sum_probs = strategy_probs[0] + strategy_probs[1] + strategy_probs[2];
    double inv_sum = 1.0 / sum_probs;
    for (int i = 0; i < STRATEGY_COUNT; i++) {
        strategy_probs[i] *= inv_sum;
        strategy_success[i] *= 0.9;
    }
}

// Simulate penguin movement in huddle
void huddle_movement(Optimizer *opt, int index, int t, ObjectiveFunction objective_function, double *workspace) {
    static double strategy_probs[STRATEGY_COUNT] = {0.3333333333333333, 0.3333333333333333, 0.3333333333333333};
    static double strategy_success[STRATEGY_COUNT] = {0.0, 0.0, 0.0};
    
    double T_prime, R;
    compute_huddle_boundary(&T_prime, &R, t, opt->max_iter);

    // Select adaptation strategy (branchless)
    double r = rand_double(0.0, 1.0);
    int strategy = (r < strategy_probs[0]) ? 0 : ((r < strategy_probs[0] + strategy_probs[1]) ? 1 : 2);

    double f, l, m_param;
    double t_norm = (double)t / opt->max_iter;
    adapt_parameters(&f, &l, &m_param, strategy, t_norm, t);

    // Compute social forces
    double S = m_param * exp(-t_norm / l) - exp(-t_norm);
    double *new_position = workspace;
    double *current_pos = opt->population[index].position;
    const double *best_pos = opt->best_solution.position;
    const double *bounds = opt->bounds;

    // Vectorized position update (SSE4.2)
    int j = 0;
    for (; j <= opt->dim - 4; j += 4) {
        __m128d current = _mm_loadu_pd(current_pos + j);
        __m128d best = _mm_loadu_pd(best_pos + j);
        __m128d rand_f = _mm_set1_pd(f * rand_double(0.0, 1.0));
        __m128d D = _mm_sub_pd(_mm_mul_pd(rand_f, best), current);
        D = _mm_max_pd(D, _mm_sub_pd(_mm_setzero_pd(), D)); // fabs
        __m128d rand_s = _mm_set1_pd(S * rand_double(0.0, 1.0));
        __m128d new_pos = _mm_add_pd(current, _mm_mul_pd(D, rand_s));
        __m128d min_bound = _mm_loadu_pd(bounds + 2 * j);
        __m128d max_bound = _mm_loadu_pd(bounds + 2 * j + 2);
        new_pos = _mm_max_pd(new_pos, min_bound);
        new_pos = _mm_min_pd(new_pos, max_bound);
        _mm_storeu_pd(new_position + j, new_pos);
        _mm_storeu_pd(current_pos + j, new_pos); // Store immediately for fitness
    }
    // Handle remaining dimensions
    for (; j < opt->dim; j++) {
        double D = fabs(f * rand_double(0.0, 1.0) * best_pos[j] - current_pos[j]);
        new_position[j] = current_pos[j] + S * D * rand_double(0.0, 1.0);
        new_position[j] = new_position[j] < bounds[2 * j] ? bounds[2 * j] :
                          new_position[j] > bounds[2 * j + 1] ? bounds[2 * j + 1] : new_position[j];
        current_pos[j] = new_position[j];
    }

    // Evaluate new solution
    double new_fitness = objective_function(current_pos);
    if (new_fitness < opt->population[index].fitness) {
        opt->population[index].fitness = new_fitness;
        strategy_success[strategy] += 1.0;
    } else {
        // Restore original position if not improved
        for (j = 0; j < opt->dim; j++) {
            current_pos[j] = new_position[j];
        }
    }

    // Update strategy probabilities periodically
    if (t % ADAPTATION_INTERVAL == 0 && t > 0) {
        update_strategy_probabilities(strategy_probs, strategy_success);
    }
}

// Main Optimization Function
void EPO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_penguins(opt);

    // Preallocate workspace (stack-based for small dim)
    double workspace[1024]; // Assumes dim <= 1024; adjust if needed
    if (opt->dim > 1024) {
        fprintf(stderr, "EPO_optimize: dim > 1024 not supported\n");
        return;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate unevaluated penguins
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == INFINITY) {
                opt->population[i].fitness = objective_function(opt->population[i].position);
            }
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Update penguin positions
        for (int i = 0; i < opt->population_size; i++) {
            huddle_movement(opt, i, iter, objective_function, workspace);
        }

        enforce_bound_constraints(opt);
    }
}
