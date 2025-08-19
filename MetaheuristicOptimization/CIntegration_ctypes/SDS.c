#include "SDS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Fast random number generator
unsigned int fast_rand_seed = 0;
double rand_double_kca(double min, double max) {
    if (!fast_rand_seed) {
        fast_rand_seed = (unsigned int)time(NULL);
    }
    fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
    return min + (max - min) * ((double)(fast_rand_seed & 0x7FFFFFFF) / (double)0x7FFFFFFF);
}

// Gaussian random number (Box-Muller transform)
double rand_normal_kca(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }
    has_spare = 1;
    double u = rand_double_kca(0.0, 1.0);
    double v = rand_double_kca(0.0, 1.0);
    double s = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    spare = sqrt(-2.0 * log(u)) * sin(2.0 * M_PI * v);
    return mean + stddev * s;
}

// Initialize agents
void initialize_agents(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = lb + (ub - lb) * rand_double_kca(0.0, 1.0);
        }
        opt->population[i].fitness = INFINITY; // Mark for evaluation
    }
}

// Evaluate a single component (simplified, assumes objective_function returns scalar)
int evaluate_component(double (*objective_function)(double *), double *hypothesis, int component_idx) {
    double value = objective_function(hypothesis);
    double max_value = fmax(1.0, fabs(value));
    double t = fabs(value) / max_value;
    return (rand_double_kca(0.0, 1.0) < t) ? 0 : 1;
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
void diffusion_phase(Optimizer *opt, int *activities) {
    double *new_hypothesis = (double *)malloc(opt->dim * sizeof(double));
    double *offset = (double *)malloc(opt->dim * sizeof(double));

    for (int i = 0; i < opt->population_size; i++) {
        if (!activities[i]) {
            int agent2_idx = fast_rand_seed % opt->population_size;
            fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
            if (activities[agent2_idx]) {
                // Copy active agent's hypothesis
                memcpy(new_hypothesis, opt->population[agent2_idx].position, opt->dim * sizeof(double));
                double r = rand_double_kca(0.0, 1.0);
                if (r < SDS_MUTATION_RATE) {
                    for (int j = 0; j < opt->dim; j++) {
                        offset[j] = rand_normal_kca(0.0, 1.0) / SDS_MUTATION_SCALE;
                        new_hypothesis[j] += offset[j];
                        double lb = opt->bounds[2 * j];
                        double ub = opt->bounds[2 * j + 1];
                        if (new_hypothesis[j] < lb) new_hypothesis[j] = lb;
                        if (new_hypothesis[j] > ub) new_hypothesis[j] = ub;
                    }
                }
                memcpy(opt->population[i].position, new_hypothesis, opt->dim * sizeof(double));
                opt->population[i].fitness = INFINITY; // Mark for re-evaluation
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    double lb = opt->bounds[2 * j];
                    double ub = opt->bounds[2 * j + 1];
                    opt->population[i].position[j] = lb + (ub - lb) * rand_double_kca(0.0, 1.0);
                }
                opt->population[i].fitness = INFINITY;
            }
        } else if (SDS_CONTEXT_SENSITIVE) {
            int agent2_idx = fast_rand_seed % opt->population_size;
            fast_rand_seed = (1664525 * fast_rand_seed + 1013904223);
            if (activities[agent2_idx]) {
                int identical = 1;
                for (int j = 0; j < opt->dim; j++) {
                    if (opt->population[agent2_idx].position[j] != opt->population[i].position[j]) {
                        identical = 0;
                        break;
                    }
                }
                if (identical) {
                    activities[i] = 0;
                    for (int j = 0; j < opt->dim; j++) {
                        double lb = opt->bounds[2 * j];
                        double ub = opt->bounds[2 * j + 1];
                        opt->population[i].position[j] = lb + (ub - lb) * rand_double_kca(0.0, 1.0);
                    }
                    opt->population[i].fitness = INFINITY;
                }
            }
        }
    }
    free(new_hypothesis);
    free(offset);
    enforce_bound_constraints(opt);
}

// Evaluate full objective function
double evaluate_full_objective(double (*objective_function)(double *), double *hypothesis) {
    return objective_function(hypothesis); // Sum of components
}

// Check Convergence
int check_convergence(Optimizer *opt) {
    if (!opt->best_solution.position) return 0;
    int cluster_size = 0;
    for (int i = 0; i < opt->population_size; i++) {
        double dist = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->population[i].position[j] - opt->best_solution.position[j];
            dist += diff * diff;
        }
        if (sqrt(dist) < SDS_CONVERGENCE_TOLERANCE) {
            cluster_size++;
        }
    }
    return ((double)cluster_size / opt->population_size) >= SDS_CLUSTER_THRESHOLD;
}

// Main Optimization Function
void SDS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate activities array
    int *activities = (int *)calloc(opt->population_size, sizeof(int));
    if (!activities) {
        fprintf(stderr, "Failed to allocate activities array\n");
        return;
    }

    initialize_agents(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        test_phase(opt, activities, objective_function);
        diffusion_phase(opt, activities);

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
        if (check_convergence(opt)) {
            printf("Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    // Clean up
    free(activities);
}
