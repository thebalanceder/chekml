#include "SDS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Fast random number generator
unsigned int fast_rand_seed = 0;
double rand_double_sds(double min, double max) {
    fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
    return min + (max - min) * ((double)(fast_rand_seed & 0x7FFFFFFF) / (double)0x7FFFFFFF);
}

// Gaussian random number (Box-Muller transform, optimized)
double rand_normal(void) {
    static int has_spare = 0;
    static double spare = 0.0;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    double u = rand_double_sds(0.0, 1.0);
    double v = rand_double_sds(0.0, 1.0);
    double s = sqrt(-2.0 * log(u));
    spare = s * sin(2.0 * M_PI * v);
    return s * cos(2.0 * M_PI * v);
}

// Static buffers for diffusion phase
static double *new_hypothesis_buffer = NULL;
static double *offset_buffer = NULL;
static int buffer_size = 0;

// Initialize agents
void initialize_agents(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            pos[j] = lb + (ub - lb) * rand_double_sds(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY;
    }
}

// Evaluate a single component
int evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx) {
    double value = objective_function(hypothesis);
    double max_value = fmax(1.0, fabs(value));
    double t = fabs(value) / max_value;
    return (rand_double_sds(0.0, 1.0) < t) ? 0 : 1;
}

// Test Phase
void test_phase(Optimizer *opt, int *activities, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        int component_idx = fast_rand_seed % SDS_MAX_COMPONENTS;
        fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
        activities[i] = evaluate_component(objective_function, opt->population[i].position, component_idx);
    }
}

// Diffusion Phase
void diffusion_phase(Optimizer *opt, int *activities, int iter, int max_iter) {
    // Resize static buffers if needed
    if (buffer_size < opt->dim) {
        buffer_size = opt->dim;
        new_hypothesis_buffer = (double *)realloc(new_hypothesis_buffer, buffer_size * sizeof(double));
        offset_buffer = (double *)realloc(offset_buffer, buffer_size * sizeof(double));
        if (!new_hypothesis_buffer || !offset_buffer) {
            fprintf(stderr, "Failed to allocate diffusion buffers\n");
            return;
        }
    }

    // Adaptive mutation rate
    double mutation_rate = SDS_MUTATION_RATE * (1.0 - 0.5 * ((double)iter / max_iter));

    double *new_hypothesis = new_hypothesis_buffer;
    double *offset = offset_buffer;

    for (int i = 0; i < opt->population_size; i++) {
        if (!activities[i]) {
            int agent2_idx = fast_rand_seed % opt->population_size;
            fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
            if (activities[agent2_idx]) {
                memcpy(new_hypothesis, opt->population[agent2_idx].position, opt->dim * sizeof(double));
                if (rand_double_sds(0.0, 1.0) < mutation_rate) {
                    double *pos = new_hypothesis;
                    const double *lb = opt->bounds;
                    const double *ub = opt->bounds + 1;
                    for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
                        offset[j] = rand_normal() * SDS_INV_MUTATION_SCALE;
                        pos[j] += offset[j];
                        if (pos[j] < *lb) pos[j] = *lb;
                        if (pos[j] > *ub) pos[j] = *ub;
                    }
                }
                memcpy(opt->population[i].position, new_hypothesis, opt->dim * sizeof(double));
                opt->population[i].fitness = INFINITY;
            } else {
                double *pos = opt->population[i].position;
                const double *lb = opt->bounds;
                const double *ub = opt->bounds + 1;
                for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
                    pos[j] = *lb + (*ub - *lb) * rand_double_sds(0.0, 1.0);
                }
                opt->population[i].fitness = INFINITY;
            }
        } else if (SDS_CONTEXT_SENSITIVE) {
            int agent2_idx = fast_rand_seed % opt->population_size;
            fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
            if (activities[agent2_idx]) {
                int identical = 1;
                const double *pos1 = opt->population[i].position;
                const double *pos2 = opt->population[agent2_idx].position;
                for (int j = 0; j < opt->dim; j++) {
                    if (pos1[j] != pos2[j]) {
                        identical = 0;
                        break;
                    }
                }
                if (identical) {
                    activities[i] = 0;
                    double *pos = opt->population[i].position;
                    const double *lb = opt->bounds;
                    const double *ub = opt->bounds + 1;
                    for (int j = 0; j < opt->dim; j++, lb += 2, ub += 2) {
                        pos[j] = *lb + (*ub - *lb) * rand_double_sds(0.0, 1.0);
                    }
                    opt->population[i].fitness = INFINITY;
                }
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Evaluate full objective function
double evaluate_full_objective(double (*objective_function)(double *), double *hypothesis) {
    return objective_function(hypothesis);
}

// Check Convergence (optimized with stagnation check)
int check_convergence(Optimizer *opt, double prev_best_fitness) {
    if (!opt->best_solution.position) return 0;
    int cluster_size = 0;
    const double *best_pos = opt->best_solution.position;
    for (int i = 0; i < opt->population_size; i++) {
        double dist_sq = 0.0;
        const double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            double diff = pos[j] - best_pos[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < SDS_CONVERGENCE_TOLERANCE * SDS_CONVERGENCE_TOLERANCE) {
            cluster_size++;
        }
    }
    int converged = ((double)cluster_size / opt->population_size) >= SDS_CLUSTER_THRESHOLD;
    converged |= (prev_best_fitness - opt->best_solution.fitness < 1e-6); // Stagnation check
    return converged;
}

// Main Optimization Function
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate activities array
    int *activities = (int *)calloc(opt->population_size, sizeof(int));
    if (!activities) {
        fprintf(stderr, "Failed to allocate activities array\n");
        return;
    }

    // Initialize random seed
    if (!fast_rand_seed) {
        fast_rand_seed = (unsigned int)time(NULL);
    }

    initialize_agents(opt);

    double prev_best_fitness = INFINITY;
    for (int iter = 0; iter < opt->max_iter; iter++) {
        test_phase(opt, activities, objective_function);
        diffusion_phase(opt, activities, iter, opt->max_iter);

        // Evaluate fitness and update best solution
        double min_fitness = INFINITY;
        int min_idx = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness == INFINITY) {
                opt->population[i].fitness = evaluate_full_objective(objective_function, opt->population[i].position);
            }
            if (opt->population[i].fitness < min_fitness) {
                min_fitness = opt->population[i].fitness;
                min_idx = i;
            }
        }
        if (min_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = min_fitness;
            memcpy(opt->best_solution.position, opt->population[min_idx].position, opt->dim * sizeof(double));
        }

        // Check convergence
        if (check_convergence(opt, prev_best_fitness)) {
            printf("Converged at iteration %d\n", iter + 1);
            break;
        }
        prev_best_fitness = opt->best_solution.fitness;
    }

    // Clean up
    free(activities);
}
