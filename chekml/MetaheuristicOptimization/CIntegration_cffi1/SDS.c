#include "SDS.h"
#include "generaloptimizer.h"
#include <stdint.h>
#include <time.h>
#include <string.h>

// Fast XOR-shift RNG
static uint64_t rng_state = 1;
double sds_rand_double(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((double)(rng_state & 0x7FFFFFFF) / (double)0x7FFFFFFF);
}

// Fast Gaussian random (Ziggurat-like approximation)
double sds_rand_normal(void) {
    static int has_spare = 0;
    static double spare = 0.0;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    double u = sds_rand_double();
    double v = sds_rand_double();
    // Simplified Box-Muller for speed with -ffast-math
    double s = sqrt(-2.0 * log(u));
    spare = s * sin(2.0 * M_PI * v);
    return s * cos(2.0 * M_PI * v);
}

// Static buffers (aligned to 64 bytes for cache)
static double *new_hypothesis_buffer __attribute__((aligned(64))) = NULL;
static double *offset_buffer __attribute__((aligned(64))) = NULL;
static int buffer_size = 0;
static int *activities_buffer __attribute__((aligned(64))) = NULL;
static int activities_size = 0;

// Initialize agents
void sds_initialize_agents(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        const double *lb = opt->bounds;
        const double *ub = opt->bounds + 1;
        for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
            pos[j] = *lb + (*ub - *lb) * sds_rand_double();
        }
        opt->population[i].fitness = INFINITY;
    }
}

// Test Phase
void sds_test_phase(Optimizer *opt, int *activities, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        int component_idx = (rng_state % SDS_MAX_COMPONENTS);
        rng_state ^= rng_state >> 27;
        activities[i] = sds_evaluate_component(objective_function, opt->population[i].position, component_idx);
    }
}

// Diffusion Phase
void sds_diffusion_phase(Optimizer *opt, int *activities, int iter) {
    // Resize buffers if needed
    if (buffer_size < opt->dim) {
        buffer_size = opt->dim;
        new_hypothesis_buffer = (double *)realloc(new_hypothesis_buffer, buffer_size * sizeof(double));
        offset_buffer = (double *)realloc(offset_buffer, buffer_size * sizeof(double));
        if (!new_hypothesis_buffer || !offset_buffer) return;
    }

    // Adaptive mutation rate
    double mutation_rate = SDS_MUTATION_RATE * (1.0 - 0.7 * ((double)iter / opt->max_iter));

    double *new_hypothesis = new_hypothesis_buffer;
    double *offset = offset_buffer;

    for (int i = 0; i < opt->population_size; i++) {
        if (!activities[i]) {
            int agent2_idx = (rng_state % opt->population_size);
            rng_state ^= rng_state >> 27;
            if (activities[agent2_idx]) {
                double *src = opt->population[agent2_idx].position;
                double *dst = new_hypothesis;
                // Unroll copy for small dimensions
                for (int j = 0; j < opt->dim; j++) {
                    dst[j] = src[j];
                }
                if (sds_rand_double() < mutation_rate) {
                    double *pos = new_hypothesis;
                    const double *lb = opt->bounds;
                    const double *ub = opt->bounds + 1;
                    // Unroll for small dims
                    for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
                        offset[j] = sds_rand_normal() * SDS_INV_MUTATION_SCALE;
                        pos[j] += offset[j];
                        pos[j] = pos[j] < *lb ? *lb : pos[j] > *ub ? *ub : pos[j];
                    }
                }
                double *target = opt->population[i].position;
                for (int j = 0; j < opt->dim; j++) {
                    target[j] = new_hypothesis[j];
                }
                opt->population[i].fitness = INFINITY;
            } else {
                double *pos = opt->population[i].position;
                const double *lb = opt->bounds;
                const double *ub = opt->bounds + 1;
                for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
                    pos[j] = *lb + (*ub - *lb) * sds_rand_double();
                }
                opt->population[i].fitness = INFINITY;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Check Convergence
int sds_check_convergence(Optimizer *opt, double prev_best_fitness) {
    if (!opt->best_solution.position) return 0;
    int cluster_size = 0;
    const double *best_pos = opt->best_solution.position;
    for (int i = 0; i < opt->population_size; i++) {
        double dist_sq = 0.0;
        const double *pos = opt->population[i].position;
        // Unroll for small dims
        for (int j = 0; j < opt->dim; j++) {
            double diff = pos[j] - best_pos[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < SDS_CONVERGENCE_TOL_SQ) {
            cluster_size++;
        }
    }
    int converged = ((double)cluster_size / opt->population_size) >= SDS_CLUSTER_THRESHOLD;
    converged |= (prev_best_fitness - opt->best_solution.fitness < SDS_STAGNATION_TOL);
    return converged;
}

// Main Optimization Function
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Resize activities buffer
    if (activities_size < opt->population_size) {
        activities_size = opt->population_size;
        activities_buffer = (int *)realloc(activities_buffer, activities_size * sizeof(int));
        if (!activities_buffer) return;
    }
    int *activities = activities_buffer;
    memset(activities, 0, opt->population_size * sizeof(int));

    // Initialize RNG
    rng_state = (uint64_t)time(NULL) | 1;

    sds_initialize_agents(opt);

    double prev_best_fitness = INFINITY;
    for (int iter = 0; iter < opt->max_iter; iter++) {
        sds_test_phase(opt, activities, objective_function);
        sds_diffusion_phase(opt, activities, iter);

        // Batch fitness evaluation
        double min_fitness = INFINITY;
        int min_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == INFINITY) {
                opt->population[i].fitness = sds_evaluate_full_objective(objective_function, opt->population[i].position);
            }
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            double *dst = opt->best_solution.position;
            double *src = opt->population[min_idx].position;
            for (int j = 0; j < opt->dim; j++) {
                dst[j] = src[j];
            }
        }

        // Check convergence
        if (sds_check_convergence(opt, prev_best_fitness)) {
            break;
        }
        prev_best_fitness = opt->best_solution.fitness;
    }
}
