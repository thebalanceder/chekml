#include "PVS.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>

#define M_PI 3.14159265358979323846

// Initialize lookup tables and PVS data
void initialize_vortex(Optimizer *opt, PVSData *data) {
    int i, j;
    double x = X_GAMMA;

    // Allocate PVS data (aligned)
    data->center = (double *)aligned_alloc(CACHE_LINE, opt->dim * sizeof(double));
    data->obj_vals = (double *)aligned_alloc(CACHE_LINE, opt->population_size * sizeof(double));
    data->prob = (double *)aligned_alloc(CACHE_LINE, opt->population_size * sizeof(double));
    data->sol = (double *)aligned_alloc(CACHE_LINE, opt->dim * sizeof(double));
    data->mutated = (double *)aligned_alloc(CACHE_LINE, opt->dim * sizeof(double));
    data->bound_diffs = (double *)aligned_alloc(CACHE_LINE, opt->dim * sizeof(double));
    data->prob_mut = 1.0 / opt->dim;
    data->prob_cross = 1.0 / opt->dim;
    data->rng_state = (uint64_t)time(NULL);

    // Precompute bound differences and center
    #pragma omp simd
    for (j = 0; j < opt->dim; j++) {
        data->bound_diffs[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
        data->center[j] = 0.5 * (opt->bounds[2 * j] + opt->bounds[2 * j + 1]);
    }
    opt->best_solution.fitness = INFINITY;

    // Initialize normal lookup table (Box-Muller)
    for (i = 0; i < LOOKUP_SIZE; i++) {
        double u1 = (i + 0.5) / LOOKUP_SIZE;
        double u2 = fast_rand_double_pvs(&data->rng_state, 0.0, 1.0);
        data->normal_lookup[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    // Initialize gamma inverse lookup table (simplified linear approximation)
    for (i = 0; i < LOOKUP_SIZE; i++) {
        double a = 0.1 + (i / (double)LOOKUP_SIZE) * 0.9;
        data->gamma_lookup[i] = 1.0 / (x * (0.5 + 0.5 * a));
    }

    // Initialize candidates
    for (i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        #pragma omp simd
        for (j = 0; j < opt->dim; j++) {
            double radius = data->gamma_lookup[0] * data->bound_diffs[j] / 2.0;
            pos[j] = data->center[j] + radius * data->normal_lookup[i % LOOKUP_SIZE];
        }
    }
    enforce_bound_constraints(opt);
}

// First Phase: Generate candidate solutions
void first_phase(Optimizer *opt, PVSData *data, int iteration, double radius) {
    int i, j, size = (iteration == 0) ? opt->population_size : opt->population_size / 2;
    
    for (i = 0; i < size; i++) {
        double *pos = opt->population[i].position;
        #pragma omp simd
        for (j = 0; j < opt->dim; j++) {
            pos[j] = data->center[j] + radius * data->normal_lookup[(i + j) % LOOKUP_SIZE];
        }
    }
    enforce_bound_constraints(opt);
}

// Polynomial Mutation
void polynomial_mutation(Optimizer *opt, PVSData *data, double *__restrict solution, double *__restrict mutated, int *__restrict state) {
    int i;
    double mut_pow = 1.0 / (1.0 + DISTRIBUTION_INDEX);
    *state = 0;

    #pragma omp simd
    for (i = 0; i < opt->dim; i++) {
        mutated[i] = solution[i];
        double r = fast_rand_double_pvs(&data->rng_state, 0.0, 1.0);
        if (r <= data->prob_mut) {
            double y = solution[i];
            double yL = opt->bounds[2 * i];
            double yU = opt->bounds[2 * i + 1];
            double delta1 = (y - yL) / data->bound_diffs[i];
            double delta2 = (yU - y) / data->bound_diffs[i];
            double rnd = fast_rand_double_pvs(&data->rng_state, 0.0, 1.0);
            double xy, val, deltaq;

            xy = (rnd <= 0.5) ? 1.0 - delta1 : 1.0 - delta2;
            val = (rnd <= 0.5) ? 2.0 * rnd + (1.0 - 2.0 * rnd) * pow(xy, DISTRIBUTION_INDEX + 1)
                               : 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * pow(xy, DISTRIBUTION_INDEX + 1);
            deltaq = (rnd <= 0.5) ? pow(val, mut_pow) - 1.0 : 1.0 - pow(val, mut_pow);

            y = y + deltaq * data->bound_diffs[i];
            mutated[i] = (y < yL) ? yL : (y > yU) ? yU : y;
            (*state)++;
        }
    }
}

// Second Phase: Crossover and Mutation
void second_phase(Optimizer *opt, PVSData *data, int iteration, ObjectiveFunction objective_function) {
    int i, j, neighbor;
    double *restrict obj_vals = data->obj_vals;
    double *restrict prob = data->prob;
    double *restrict sol = data->sol;
    double *restrict mutated = data->mutated;

    // Update fitness values
    #pragma omp simd
    for (i = 0; i < opt->population_size; i++) {
        obj_vals[i] = opt->population[i].fitness;
    }

    // Compute roulette wheel probabilities
    double max_val = obj_vals[0];
    for (i = 1; i < opt->population_size; i++) {
        if (obj_vals[i] > max_val) max_val = obj_vals[i];
    }
    double sum_prob = 0.0;
    #pragma omp simd
    for (i = 0; i < opt->population_size; i++) {
        prob[i] = 0.9 * (max_val - obj_vals[i]) + 0.1;
        sum_prob += prob[i];
    }
    double inv_sum = 1.0 / sum_prob;
    prob[0] *= inv_sum;
    #pragma omp simd
    for (i = 1; i < opt->population_size; i++) {
        prob[i] = prob[i - 1] + (prob[i] * inv_sum);
    }

    // Process second half of population
    for (i = opt->population_size / 2; i < opt->population_size; i++) {
        // Roulette wheel selection
        double r = fast_rand_double_pvs(&data->rng_state, 0.0, 1.0);
        neighbor = opt->population_size - 1;
        for (j = 0; j < opt->population_size - 1; j++) {
            if (r <= prob[j]) {
                neighbor = j;
                break;
            }
        }
        while (i == neighbor) {
            r = fast_rand_double_pvs(&data->rng_state, 0.0, 1.0);
            neighbor = opt->population_size - 1;
            for (j = 0; j < opt->population_size - 1; j++) {
                if (r <= prob[j]) {
                    neighbor = j;
                    break;
                }
            }
        }

        // Crossover
        double *pop_i = opt->population[i].position;
        double *pop_n = opt->population[neighbor].position;
        int param2change = fast_rng(&data->rng_state) % opt->dim;
        #pragma omp simd
        for (j = 0; j < opt->dim; j++) {
            sol[j] = pop_i[j];
            double do_cross = (fast_rand_double_pvs(&data->rng_state, 0.0, 1.0) < data->prob_cross) || (j == param2change);
            sol[j] += do_cross * (pop_i[j] - pop_n[j]) * (fast_rand_double_pvs(&data->rng_state, 0.0, 1.0) - 0.5) * 2.0;
            sol[j] = (sol[j] < opt->bounds[2 * j]) ? opt->bounds[2 * j] :
                     (sol[j] > opt->bounds[2 * j + 1]) ? opt->bounds[2 * j + 1] : sol[j];
        }

        // Evaluate new solution
        double obj_val_sol = objective_function(sol);
        if (obj_val_sol < obj_vals[i]) {
            memcpy(pop_i, sol, opt->dim * sizeof(double));
            opt->population[i].fitness = obj_val_sol;
            obj_vals[i] = obj_val_sol;
        } else {
            int state;
            polynomial_mutation(opt, data, pop_i, mutated, &state);
            if (state > 0) {
                double obj_val_mut = objective_function(mutated);
                if (obj_val_mut < obj_vals[i]) {
                    memcpy(pop_i, mutated, opt->dim * sizeof(double));
                    opt->population[i].fitness = obj_val_mut;
                    obj_vals[i] = obj_val_mut;
                }
            }
        }
    }
}

// Main Optimization Function
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    int iteration = 0;
    double x = X_GAMMA;
    PVSData data = {0};

    initialize_vortex(opt, &data);

    while (iteration < opt->max_iter) {
        // Update radius
        double a = (opt->max_iter - iteration) / (double)opt->max_iter;
        a = a > 0.1 ? a : 0.1;
        int gamma_idx = (int)((a - 0.1) * LOOKUP_SIZE / 0.9);
        double radius = data.gamma_lookup[gamma_idx] * data.bound_diffs[0] / 2.0;

        // First phase
        first_phase(opt, &data, iteration, radius);
        int size = (iteration == 0) ? opt->population_size : opt->population_size / 2;
        for (int i = 0; i < size; i++) {
            double fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = fitness;
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                memcpy(data.center, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Second phase
        second_phase(opt, &data, iteration, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                memcpy(data.center, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        iteration++;
    }

    // Cleanup
    free(data.center);
    free(data.obj_vals);
    free(data.prob);
    free(data.sol);
    free(data.mutated);
    free(data.bound_diffs);
}
