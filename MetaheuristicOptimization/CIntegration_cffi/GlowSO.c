#include "GlowSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Inline function to convert objective function value to minimization form
static inline double convert_to_min(double fcn) {
    return (fcn >= 0) ? 1.0 / (1.0 + fcn) : 1.0 + fabs(fcn);
}

// Inline function to compute Euclidean distance between two positions
static inline double euclidean_distance(double *pos1, double *pos2, int dim) {
    double dist = 0.0;
    for (int j = 0; j < dim; j++) {
        double diff = pos1[j] - pos2[j];
        dist += diff * diff;
    }
    return sqrt(dist);
}

// Select neighbor using roulette wheel selection
static inline int select_by_roulette(double *probs, int size) {
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
void luciferin_update(Optimizer *opt, double (*objective_function)(double *)) {
    Solution *pop = opt->population;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = convert_to_min(objective_function(pop[i].position));
        pop[i].fitness = (1.0 - LUCIFERIN_DECAY) * pop[i].fitness +
                         LUCIFERIN_ENHANCEMENT * fitness;
    }
}

// Movement Phase
void movement_phase(Optimizer *opt, double *decision_range, double *distances, int *neighbors, double *probs, double *current_pos) {
    Solution *pop = opt->population;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        // Copy current position
        double *pos = pop[i].position;
        for (int j = 0; j < dim; j++) {
            current_pos[j] = pos[j];
        }
        double current_luciferin = pop[i].fitness;

        // Find neighbors
        int neighbor_count = 0;
        for (int j = 0; j < pop_size; j++) {
            if (distances[i * pop_size + j] < decision_range[i] && pop[j].fitness > current_luciferin) {
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
        if (prob_sum > 0.0) {
            for (int k = 0; k < neighbor_count; k++) {
                probs[k] /= prob_sum;
            }
        } else {
            continue;
        }

        // Select neighbor and move
        int selected_idx = neighbors[select_by_roulette(probs, neighbor_count)];
        double *selected_pos = pop[selected_idx].position;
        double distance = distances[i * pop_size + selected_idx];

        if (distance > 0.0) {
            for (int j = 0; j < dim; j++) {
                pos[j] = current_pos[j] + GSO_STEP_SIZE * (selected_pos[j] - current_pos[j]) / distance;
            }
        }
    }

    enforce_bound_constraints(opt);
}

// Decision Range Update Phase
void decision_range_update(Optimizer *opt, double *decision_range, double *distances, int *neighbors) {
    Solution *pop = opt->population;
    int pop_size = opt->population_size;

    for (int i = 0; i < pop_size; i++) {
        double current_luciferin = pop[i].fitness;
        int neighbor_count = 0;
        for (int j = 0; j < pop_size; j++) {
            if (distances[i * pop_size + j] < decision_range[i] && pop[j].fitness > current_luciferin) {
                neighbors[neighbor_count++] = j;
            }
        }
        decision_range[i] = fmin(SENSOR_RANGE,
                                 fmax(0.0, decision_range[i] +
                                           NEIGHBOR_THRESHOLD * (NEIGHBOR_COUNT - neighbor_count)));
    }
}

// Main Optimization Function
void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate reusable arrays
    double *decision_range = (double *)malloc(pop_size * sizeof(double));
    double *distances = (double *)malloc(pop_size * pop_size * sizeof(double));
    int *neighbors = (int *)malloc(pop_size * sizeof(int));
    double *probs = (double *)malloc(pop_size * sizeof(double));
    double *current_pos = (double *)malloc(dim * sizeof(double));

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
    Solution *pop = opt->population;
    for (int i = 0; i < pop_size; i++) {
        pop[i].fitness = LUCIFERIN_INITIAL;
        decision_range[i] = DECISION_RANGE_INITIAL;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Update luciferin and find best
        luciferin_update(opt, objective_function);
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) {
                best_idx = i;
            }
        }
        double fitness = convert_to_min(objective_function(pop[best_idx].position));
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < dim; j++) {
                opt->best_solution.position[j] = pop[best_idx].position[j];
            }
        }

        // Compute distances once per iteration
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < pop_size; j++) {
                distances[i * pop_size + j] = euclidean_distance(pop[i].position, pop[j].position, dim);
            }
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
