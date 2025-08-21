#include "ACO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Fast XOR-shift random number generator
typedef struct {
    unsigned int x;
} XorShift;

inline void xorshift_init(XorShift *rng, unsigned int seed) {
    rng->x = seed ? seed : (unsigned int)time(NULL);
}

inline double xorshift_double(XorShift *rng, double min, double max) {
    rng->x ^= rng->x << 13;
    rng->x ^= rng->x >> 17;
    rng->x ^= rng->x << 5;
    return min + (max - min) * (rng->x / (double)0xFFFFFFFF);
}

// Roulette wheel selection
inline int roulette_wheel_selection(double *prob, int size, XorShift *rng) {
    double r = xorshift_double(rng, 0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += prob[i];
        if (r <= cumsum) {
            return i;
        }
    }
    return size - 1;
}

// Construct solutions for all ants
void construct_solutions(Optimizer *opt, double (*objective_function)(double *), double *tau, double *bins, int *tours, double *prob) {
    const int n_bins = ACO_N_BINS;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    #pragma omp parallel
    {
        XorShift rng;
        xorshift_init(&rng, (unsigned int)(omp_get_thread_num() + 1));

        #pragma omp for
        for (int k = 0; k < pop_size; k++) {
            int *tour = tours + k * dim;
            double *position = opt->population[k].position;

            for (int d = 0; d < dim; d++) {
                double sum_P = 0.0;
                double *tau_d = tau + d * n_bins;

                // Compute probabilities
                for (int i = 0; i < n_bins; i++) {
                    prob[i] = tau_d[i]; // Simplified: eta=1.0, ACO_ALPHA=1.0
                    sum_P += prob[i];
                }

                // Normalize probabilities
                if (sum_P > 0.0) {
                    double inv_sum_P = 1.0 / sum_P;
                    for (int i = 0; i < n_bins; i++) {
                        prob[i] *= inv_sum_P;
                    }
                }

                // Select bin
                int bin_idx = roulette_wheel_selection(prob, n_bins, &rng);
                tour[d] = bin_idx;
                position[d] = bins[d * n_bins + bin_idx];
            }

            opt->population[k].fitness = objective_function(position);
        }
    }

    enforce_bound_constraints(opt);
}

// Update pheromone trails using precomputed tours
void update_pheromones(Optimizer *opt, double *tau, int *tours) {
    const int n_bins = ACO_N_BINS;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;
    const double best_fitness = opt->best_solution.fitness;

    for (int k = 0; k < pop_size; k++) {
        double delta = ACO_Q / (1.0 + (opt->population[k].fitness > best_fitness ? 
                                      opt->population[k].fitness - best_fitness : 0.0));
        int *tour = tours + k * dim;
        for (int d = 0; d < dim; d++) {
            tau[d * n_bins + tour[d]] += delta;
        }
    }
}

// Apply pheromone evaporation
void evaporate_pheromones(double *tau, int dim) {
    const int total_size = ACO_N_BINS * dim;
    for (int i = 0; i < total_size; i++) {
        tau[i] *= ACO_INV_RHO;
    }
}

// Main Optimization Function
void ACO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate aligned memory for cache efficiency
    const int n_bins = ACO_N_BINS;
    const int dim = opt->dim;
    const int pop_size = ACO_N_ANT;
    const int tau_size = n_bins * dim;
    const int bins_size = n_bins * dim;
    const int tours_size = pop_size * dim;

    double *tau = (double *)aligned_alloc(64, tau_size * sizeof(double));
    double *bins = (double *)aligned_alloc(64, bins_size * sizeof(double));
    int *tours = (int *)aligned_alloc(64, tours_size * sizeof(int));
    double *prob = (double *)malloc(n_bins * sizeof(double));

    // Initialize tau and bins
    for (int d = 0; d < dim; d++) {
        double lower = opt->bounds[2 * d];
        double range = opt->bounds[2 * d + 1] - lower;
        double bin_step = range / (n_bins - 1);
        for (int i = 0; i < n_bins; i++) {
            tau[d * n_bins + i] = ACO_TAU0;
            bins[d * n_bins + i] = lower + i * bin_step;
        }
    }

    // Set population size and max iterations
    opt->population_size = pop_size;
    opt->max_iter = ACO_MAX_ITER;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        construct_solutions(opt, objective_function, tau, bins, tours, prob);
        update_pheromones(opt, tau, tours);
        evaporate_pheromones(tau, dim);

        // Find best solution
        double current_best_fitness = INFINITY;
        int current_best_idx = 0;
        for (int k = 0; k < pop_size; k++) {
            if (opt->population[k].fitness < current_best_fitness) {
                current_best_fitness = opt->population[k].fitness;
                current_best_idx = k;
            }
        }

        // Update global best
        if (current_best_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = current_best_fitness;
            for (int d = 0; d < dim; d++) {
                opt->best_solution.position[d] = opt->population[current_best_idx].position[d];
            }
        }

        printf("Iteration %d: Best Cost = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Free memory
    free(tau);
    free(bins);
    free(tours);
    free(prob);
}
