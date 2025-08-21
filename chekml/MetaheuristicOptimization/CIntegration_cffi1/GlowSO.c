#include "GlowSO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Inline function to convert objective function value to minimization form
static inline double convert_to_min(double fcn) {
    return (fcn >= 0) ? 1.0 / (1.0 + fcn) : 1.0 + fabs(fcn);
}

// Inline function to compute squared Euclidean distance (avoid sqrt for neighbor checks)
static inline double squared_distance(const double *restrict pos1, const double *restrict pos2, int dim) {
    double dist = 0.0;
    for (int j = 0; j < dim; j++) {
        double diff = pos1[j] - pos2[j];
        dist += diff * diff;
    }
    return dist;
}

// Inline function to compute Euclidean distance (used only when needed)
static inline double euclidean_distance(const double *restrict pos1, const double *restrict pos2, int dim) {
    return sqrt(squared_distance(pos1, pos2, dim));
}

// Select neighbor using roulette wheel selection
static inline int select_by_roulette(const double *restrict probs, int size) {
    double cum_prob = 0.0;
    double rn = gso_rand_double(0.0, 1.0);
    for (int i = 0; i < size; i++) {
        cum_prob += probs[i];
        if (cum_prob >= rn) {
            return i;
        }
    }
    return size - 1;
}

// Luciferin Update Phase
void luciferin_update(Optimizer *restrict opt, double (*objective_function)(double *)) {
    Solution *restrict pop = opt->population;
    const int pop_size = opt->population_size;
    const double decay_factor = 1.0 - LUCIFERIN_DECAY;

    for (int i = 0; i < pop_size; i++) {
        double fitness = convert_to_min(objective_function(pop[i].position));
        pop[i].fitness = decay_factor * pop[i].fitness + LUCIFERIN_ENHANCEMENT * fitness;
    }
}

// Movement Phase
void movement_phase(Optimizer *restrict opt, double *restrict decision_range, double *restrict distances, int *restrict neighbors, double *restrict probs, double *restrict current_pos) {
    Solution *restrict pop = opt->population;
    const int dim = opt->dim;
    const int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double *restrict pos = pop[i].position;
        const double current_luciferin = pop[i].fitness;
        const double decision_range_sq = decision_range[i] * decision_range[i];

        // Copy current position
        memcpy(current_pos, pos, dim * sizeof(double));

        // Find neighbors
        int neighbor_count = 0;
        for (int j = 0; j < pop_size; j++) {
            if (i != j && distances[i * pop_size + j] < decision_range_sq && pop[j].fitness > current_luciferin) {
                neighbors[neighbor_count++] = j;
            }
        }

        if (neighbor_count == 0) {
            continue;
        }

        // Compute probabilities
        double prob_sum = 0.0;
        for (int k = 0; k < neighbor_count; k++) {
            probs[k] = pop[neighbors[k]].fitness - current_luciferin;
            prob_sum += probs[k];
        }
        if (prob_sum <= 0.0) {
            continue;
        }
        for (int k = 0; k < neighbor_count; k++) {
            probs[k] /= prob_sum;
        }

        // Select neighbor and move
        int selected_idx = neighbors[select_by_roulette(probs, neighbor_count)];
        double *restrict selected_pos = pop[selected_idx].position;
        double distance = sqrt(distances[i * pop_size + selected_idx]);

        if (distance > 0.0) {
            const double step = GSO_STEP_SIZE / distance;
            for (int j = 0; j < dim; j++) {
                pos[j] = current_pos[j] + step * (selected_pos[j] - current_pos[j]);
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Decision Range Update Phase
void decision_range_update(Optimizer *restrict opt, double *restrict decision_range, double *restrict distances, int *restrict neighbors) {
    Solution *restrict pop = opt->population;
    const int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        const double current_luciferin = pop[i].fitness;
        const double decision_range_sq = decision_range[i] * decision_range[i];
        int neighbor_count = 0;

        for (int j = 0; j < pop_size; j++) {
            if (i != j && distances[i * pop_size + j] < decision_range_sq && pop[j].fitness > current_luciferin) {
                neighbors[neighbor_count++] = j;
            }
        }

        double new_range = decision_range[i] + NEIGHBOR_THRESHOLD * (NEIGHBOR_COUNT - neighbor_count);
        decision_range[i] = fmin(SENSOR_RANGE, fmax(0.0, new_range));
    }
}

// Main Optimization Function
void GlowSO_optimize(Optimizer *restrict opt, double (*objective_function)(double *)) {
    const int pop_size = opt->population_size;
    const int dim = opt->dim;
    srand((unsigned int)time(NULL));

    // Allocate reusable arrays
    double *restrict decision_range = (double *)malloc(pop_size * sizeof(double));
    double *restrict distances = (double *)malloc(pop_size * pop_size * sizeof(double));
    int *restrict neighbors = (int *)malloc(pop_size * sizeof(int));
    double *restrict probs = (double *)malloc(pop_size * sizeof(double));
    double *restrict current_pos = (double *)malloc(dim * sizeof(double));

    if (!decision_range || !distances || !neighbors || !probs || !current_pos) {
        fprintf(stderr, "Memory allocation failed\n");
        free(decision_range);
        free(distances);
        free(neighbors);
        free(probs);
        free(current_pos);
        return;
    }

    // Initialize luciferin and decision range
    Solution *restrict pop = opt->population;
    for (int i = 0; i < pop_size; i++) {
        pop[i].fitness = LUCIFERIN_INITIAL;
        decision_range[i] = DECISION_RANGE_INITIAL;
    }

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update luciferin and find best
        luciferin_update(opt, objective_function);
        int best_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) {
                best_idx = i;
            }
        }
        double fitness = convert_to_min(objective_function(pop[best_idx].position));
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            memcpy(opt->best_solution.position, pop[best_idx].position, dim * sizeof(double));
        }

        // Compute squared distances once per iteration
        for (int i = 0; i < pop_size; i++) {
            const double *restrict pos_i = pop[i].position;
            for (int j = i + 1; j < pop_size; j++) {
                double dist_sq = squared_distance(pos_i, pop[j].position, dim);
                distances[i * pop_size + j] = dist_sq;
                distances[j * pop_size + i] = dist_sq; // Symmetric matrix
            }
            distances[i * pop_size + i] = 0.0;
        }

        movement_phase(opt, decision_range, distances, neighbors, probs, current_pos);
        decision_range_update(opt, decision_range, distances, neighbors);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(decision_range);
    free(distances);
    free(neighbors);
    free(probs);
    free(current_pos);
}
