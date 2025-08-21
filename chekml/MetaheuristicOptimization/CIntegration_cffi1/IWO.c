#include "IWO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Xorshift PRNG for fast uniform random numbers
static uint64_t xorshift_state = 0;

void seed_xorshift(uint64_t seed) {
    xorshift_state = seed ? seed : (uint64_t)time(NULL);
}

uint32_t xorshift32() {
    uint64_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xorshift_state = x;
    return (uint32_t)(x & 0xFFFFFFFF);
}

double rand_double_iwo(double min, double max) {
    return min + (max - min) * (xorshift32() / (double)0xFFFFFFFF);
}

// Fast Gaussian random number generator (Ziggurat-inspired approximation)
double rand_normal_iwo() {
    static const double r = 3.442619855899; // Ziggurat radius
    static const double f = 0.081679717;   // Precomputed constant
    while (1) {
        uint32_t u = xorshift32();
        double x = (int32_t)u * (2.0 / 0x7FFFFFFF) * r;
        double y = xorshift32() / (double)0xFFFFFFFF;
        if (y < f) return x;
        if (fabs(x) < r - 1.0) return x;
        // Fallback to Box-Muller for tail
        double u1 = xorshift32() / (double)0xFFFFFFFF;
        double u2 = xorshift32() / (double)0xFFFFFFFF;
        if (u1 == 0.0) u1 = 1e-10;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        return z;
    }
}

// Comparison function for qsort
typedef struct {
    int index;
    double fitness;
} IndexedFitness;

int compare_fitness_iwo(const void *a, const void *b) {
    double diff = ((IndexedFitness *)a)->fitness - ((IndexedFitness *)b)->fitness;
    return (diff > 0) - (diff < 0);
}

// Initialize Population
void initialize_population_iwo(Optimizer *opt) {
    opt->population_size = INITIAL_POP_SIZE;
    opt->population = (Solution *)malloc(MAX_POP_SIZE * sizeof(Solution));
    
    for (int i = 0; i < MAX_POP_SIZE; i++) {
        opt->population[i].position = (double *)aligned_alloc(64, opt->dim * sizeof(double));
        opt->population[i].fitness = INFINITY;
        if (i < INITIAL_POP_SIZE) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = rand_double_iwo(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
            }
        }
        printf("Initialized population[%d].position = %p\n", i, (void *)opt->population[i].position);
    }
    enforce_bound_constraints(opt);
}

// Evaluate Population
void evaluate_population_iwo(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", i);
            continue;
        }
        double fitness = objective_function(opt->population[i].position);
        if (isnan(fitness)) {
            fprintf(stderr, "Warning: NaN fitness detected for population[%d]\n", i);
            fitness = INFINITY;
        }
        opt->population[i].fitness = fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            printf("New best fitness at population[%d]: %f\n", i, fitness);
        }
    }
}

// Update Standard Deviation
inline double update_standard_deviation(int iteration, int max_iter) {
    double t = (double)(max_iter - iteration) / max_iter;
    return (SIGMA_INITIAL - SIGMA_FINAL) * (t * t) + SIGMA_FINAL; // Simplified EXPONENT=2
}

// Reproduction Phase
void reproduction_phase(Optimizer *opt, double sigma, double (*objective_function)(double *)) {
    double best_cost = INFINITY;
    double worst_cost = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double f = opt->population[i].fitness;
        best_cost = f < best_cost ? f : best_cost;
        worst_cost = f > worst_cost ? f : worst_cost;
    }
    double cost_range = worst_cost - best_cost;
    int seed_diff = MAX_SEEDS - MIN_SEEDS;

    int old_pop_size = opt->population_size;
    for (int i = 0; i < old_pop_size; i++) {
        int num_seeds = MIN_SEEDS;
        if (cost_range > 1e-10) {
            double ratio = (worst_cost - opt->population[i].fitness) / cost_range;
            num_seeds += (int)(seed_diff * ratio);
        }
        printf("Solution[%d] fitness=%f, num_seeds=%d\n", i, opt->population[i].fitness, num_seeds);

        double *parent_pos = opt->population[i].position;
        for (int j = 0; j < num_seeds && opt->population_size < MAX_POP_SIZE; j++) {
            int new_idx = opt->population_size;
            if (!opt->population[new_idx].position) {
                fprintf(stderr, "Error: population[%d].position is NULL\n", new_idx);
                continue;
            }
            double *new_pos = opt->population[new_idx].position;
            if (opt->dim == 2) {
                new_pos[0] = parent_pos[0] + sigma * rand_normal_iwo();
                new_pos[0] = fmax(new_pos[0], opt->bounds[0]);
                new_pos[0] = fmin(new_pos[0], opt->bounds[1]);
                new_pos[1] = parent_pos[1] + sigma * rand_normal_iwo();
                new_pos[1] = fmax(new_pos[1], opt->bounds[2]);
                new_pos[1] = fmin(new_pos[1], opt->bounds[3]);
            } else {
                for (int k = 0; k < opt->dim; k++) {
                    new_pos[k] = parent_pos[k] + sigma * rand_normal_iwo();
                    new_pos[k] = fmax(new_pos[k], opt->bounds[2 * k]);
                    new_pos[k] = fmin(new_pos[k], opt->bounds[2 * k + 1]);
                }
            }
            double fitness = objective_function(new_pos);
            if (isnan(fitness)) {
                fprintf(stderr, "Warning: NaN fitness detected for new seed[%d]\n", new_idx);
                fitness = INFINITY;
            }
            opt->population[new_idx].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                memcpy(opt->best_solution.position, new_pos, opt->dim * sizeof(double));
                printf("New best fitness at seed[%d]: %f\n", new_idx, fitness);
            }
            opt->population_size++;
        }
    }
    enforce_bound_constraints(opt);
}

// Competitive Exclusion and Sorting
void competitive_exclusion(Optimizer *opt, double (*objective_function)(double *)) {
    IndexedFitness *indexed = (IndexedFitness *)malloc(opt->population_size * sizeof(IndexedFitness));
    for (int i = 0; i < opt->population_size; i++) {
        indexed[i].index = i;
        indexed[i].fitness = opt->population[i].fitness;
    }

    qsort(indexed, opt->population_size, sizeof(IndexedFitness), compare_fitness_iwo);

    Solution *temp_population = (Solution *)malloc(opt->population_size * sizeof(Solution));
    for (int i = 0; i < opt->population_size; i++) {
        temp_population[i] = opt->population[indexed[i].index];
    }
    memcpy(opt->population, temp_population, opt->population_size * sizeof(Solution));
    free(temp_population);
    free(indexed);

    if (opt->population_size > MAX_POP_SIZE) {
        for (int i = MAX_POP_SIZE; i < opt->population_size; i++) {
            opt->population[i].fitness = INFINITY;
        }
        opt->population_size = MAX_POP_SIZE;
    }

    double best_fitness = objective_function(opt->best_solution.position);
    if (fabs(best_fitness - opt->best_solution.fitness) > 1e-10) {
        fprintf(stderr, "Warning: Best solution fitness mismatch: stored=%f, computed=%f\n", 
                opt->best_solution.fitness, best_fitness);
        opt->best_solution.fitness = best_fitness;
    }

    for (int i = 0; i < opt->population_size; i++) {
        printf("After competitive_exclusion: population[%d].position = %p\n", i, (void *)opt->population[i].position);
    }
}

// Main Optimization Function
void IWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    static int seeded = 0;
    if (!seeded) {
        seed_xorshift((uint64_t)time(NULL));
        seeded = 1;
    }

    initialize_population_iwo(opt);
    opt->best_solution.fitness = INFINITY;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_population_iwo(opt, objective_function);
        double sigma = update_standard_deviation(iter, opt->max_iter);
        reproduction_phase(opt, sigma, objective_function);
        competitive_exclusion(opt, objective_function);
        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
