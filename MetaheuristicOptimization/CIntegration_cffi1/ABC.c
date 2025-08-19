#include "ABC.h"
#include "generaloptimizer.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// 🎲 Fast Xorshift128+ RNG
inline uint64_t xorshift128plus(XorshiftState_ABC *state) {
    uint64_t x = state->state[0];
    uint64_t y = state->state[1];
    state->state[0] = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    state->state[1] = x;
    return x + y;
}

inline double abc_rand_double(double min, double max, XorshiftState_ABC *state) {
    uint64_t r = xorshift128plus(state);
    return min + (max - min) * (r * 5.421010862427522e-20); // 1/2^64
}

// 🎲 Utility Functions
int abc_roulette_wheel_selection(double *restrict probabilities, int size, double *restrict cumsum) {
    XorshiftState_ABC state = {{0x123456789abcdefULL, 0xfedcba987654321ULL}}; // Fixed seed for simplicity
    double r = abc_rand_double(0.0, 1.0, &state);
    cumsum[0] = probabilities[0];
    for (int i = 1; i < size; i++) {
        cumsum[i] = cumsum[i - 1] + probabilities[i];
    }
    for (int i = 0; i < size; i++) {
        if (r <= cumsum[i]) {
            return i;
        }
    }
    return 0;
}

// 🐝 ABC Algorithm Phases
void employed_bee_phase(Optimizer *opt, double (*objective_function)(double *), 
                       int *restrict trial_counters, double *restrict phi, double *restrict new_position) {
    XorshiftState_ABC state = {{0x123456789abcdefULL, 0xfedcba987654321ULL}};
    for (int i = 0; i < opt->population_size; i++) {
        // Select random bee (k != i)
        int k;
        do {
            k = xorshift128plus(&state) % opt->population_size;
        } while (k == i);
        
        // Generate acceleration coefficient (SIMD for large dim)
        int j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            __m256d rand = _mm256_set1_pd(abc_rand_double(-1.0, 1.0, &state));
            _mm256_storeu_pd(&phi[j], _mm256_mul_pd(_mm256_set1_pd(ABC_ACCELERATION_BOUND), rand));
        }
        for (; j < opt->dim; j++) {
            phi[j] = ABC_ACCELERATION_BOUND * abc_rand_double(-1.0, 1.0, &state);
        }
        
        // Create new solution (SIMD)
        j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            __m256d pos_i = _mm256_loadu_pd(&opt->population[i].position[j]);
            __m256d pos_k = _mm256_loadu_pd(&opt->population[k].position[j]);
            __m256d diff = _mm256_sub_pd(pos_i, pos_k);
            __m256d update = _mm256_mul_pd(_mm256_loadu_pd(&phi[j]), diff);
            __m256d new_pos = _mm256_add_pd(pos_i, update);
            __m256d lb = _mm256_loadu_pd(&opt->bounds[2 * j]);
            __m256d ub = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
            new_pos = _mm256_max_pd(_mm256_min_pd(new_pos, ub), lb);
            _mm256_storeu_pd(&new_position[j], new_pos);
        }
        for (; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + 
                             phi[j] * (opt->population[i].position[j] - opt->population[k].position[j]);
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }
        
        // Evaluate new solution
        double new_cost = objective_function(new_position);
        
        // Greedy selection
        if (new_cost <= opt->population[i].fitness) {
            memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
            opt->population[i].fitness = new_cost;
            trial_counters[i] = 0;
        } else {
            trial_counters[i]++;
        }
    }
}

void onlooker_bee_phase(Optimizer *opt, double (*objective_function)(double *), 
                       int *restrict trial_counters, double *restrict phi, double *restrict new_position,
                       double *restrict fitness, double *restrict probabilities, double *restrict cumsum) {
    XorshiftState_ABC state = {{0x123456789abcdefULL, 0xfedcba987654321ULL}};
    
    // Calculate fitness (SIMD for large population)
    double mean_cost = 0.0;
    int i = 0;
    __m256d sum = _mm256_setzero_pd();
    for (; i <= opt->population_size - 4; i += 4) {
        sum = _mm256_add_pd(sum, _mm256_loadu_pd(&opt->population[i].fitness));
    }
    double sums[4];
    _mm256_storeu_pd(sums, sum);
    mean_cost += sums[0] + sums[1] + sums[2] + sums[3];
    for (; i < opt->population_size; i++) {
        mean_cost += opt->population[i].fitness;
    }
    mean_cost /= opt->population_size;
    
    // Fast exp approximation: exp(x) ≈ (1 + x/16)^16 for small x
    double fitness_sum = 0.0;
    for (i = 0; i < opt->population_size; i++) {
        double x = -opt->population[i].fitness / mean_cost;
        x = x / 16.0;
        double t = 1.0 + x;
        double t2 = t * t;
        double t4 = t2 * t2;
        double t8 = t4 * t4;
        fitness[i] = t8 * t8; // (1 + x/16)^16
        fitness_sum += fitness[i];
    }
    
    for (i = 0; i < opt->population_size; i++) {
        probabilities[i] = fitness[i] / fitness_sum;
    }
    
    int n_onlookers = (int)(ABC_ONLOOKER_RATIO * opt->population_size);
    for (int m = 0; m < n_onlookers; m++) {
        // Select food source
        int i = abc_roulette_wheel_selection(probabilities, opt->population_size, cumsum);
        
        // Select random bee (k != i)
        int k;
        do {
            k = xorshift128plus(&state) % opt->population_size;
        } while (k == i);
        
        // Generate acceleration coefficient (SIMD)
        int j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            __m256d rand = _mm256_set1_pd(abc_rand_double(-1.0, 1.0, &state));
            _mm256_storeu_pd(&phi[j], _mm256_mul_pd(_mm256_set1_pd(ABC_ACCELERATION_BOUND), rand));
        }
        for (; j < opt->dim; j++) {
            phi[j] = ABC_ACCELERATION_BOUND * abc_rand_double(-1.0, 1.0, &state);
        }
        
        // Create new solution (SIMD)
        j = 0;
        for (; j <= opt->dim - 4; j += 4) {
            __m256d pos_i = _mm256_loadu_pd(&opt->population[i].position[j]);
            __m256d pos_k = _mm256_loadu_pd(&opt->population[k].position[j]);
            __m256d diff = _mm256_sub_pd(pos_i, pos_k);
            __m256d update = _mm256_mul_pd(_mm256_loadu_pd(&phi[j]), diff);
            __m256d new_pos = _mm256_add_pd(pos_i, update);
            __m256d lb = _mm256_loadu_pd(&opt->bounds[2 * j]);
            __m256d ub = _mm256_loadu_pd(&opt->bounds[2 * j + 1]);
            new_pos = _mm256_max_pd(_mm256_min_pd(new_pos, ub), lb);
            _mm256_storeu_pd(&new_position[j], new_pos);
        }
        for (; j < opt->dim; j++) {
            new_position[j] = opt->population[i].position[j] + 
                             phi[j] * (opt->population[i].position[j] - opt->population[k].position[j]);
            new_position[j] = fmin(fmax(new_position[j], opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }
        
        // Evaluate new solution
        double new_cost = objective_function(new_position);
        
        // Greedy selection
        if (new_cost <= opt->population[i].fitness) {
            memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
            opt->population[i].fitness = new_cost;
            trial_counters[i] = 0;
        } else {
            trial_counters[i]++;
        }
    }
}

void scout_bee_phase(Optimizer *opt, double (*objective_function)(double *), 
                    int *restrict trial_counters) {
    XorshiftState_ABC state = {{0x123456789abcdefULL, 0xfedcba987654321ULL}};
    int trial_limit = (int)(ABC_TRIAL_LIMIT_FACTOR * opt->dim * opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        if (trial_counters[i] >= trial_limit) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = opt->bounds[2 * j] + 
                                                abc_rand_double(0.0, 1.0, &state) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
            opt->population[i].fitness = objective_function(opt->population[i].position);
            trial_counters[i] = 0;
        }
    }
}

void abc_update_best_solution(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }
}

// 🚀 Main Optimization Function
void ABC_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize trial counters
    int *trial_counters = (int *)calloc(opt->population_size, sizeof(int));
    
    // Pre-allocate temporary arrays (aligned for SIMD)
    double *phi = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
    double *new_position = (double *)_mm_malloc(opt->dim * sizeof(double), 32);
    double *fitness = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    double *probabilities = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    double *cumsum = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    
    // Initialize population
    XorshiftState_ABC state = {{0x123456789abcdefULL, 0xfedcba987654321ULL}};
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            abc_rand_double(0.0, 1.0, &state) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }
    
    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    abc_update_best_solution(opt, objective_function);
    
    // Main loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        employed_bee_phase(opt, objective_function, trial_counters, phi, new_position);
        onlooker_bee_phase(opt, objective_function, trial_counters, phi, new_position, fitness, probabilities, cumsum);
        scout_bee_phase(opt, objective_function, trial_counters);
        abc_update_best_solution(opt, objective_function);
        
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    // Clean up
    free(trial_counters);
    _mm_free(phi);
    _mm_free(new_position);
    _mm_free(fitness);
    _mm_free(probabilities);
    _mm_free(cumsum);
}
