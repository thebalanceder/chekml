#include "GlowSO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Convert objective function value to minimization form
double convert_to_min(double fcn) {
    if (fcn >= 0) {
        return 1.0 / (1.0 + fcn);
    }
    return 1.0 + fabs(fcn);
}

// Compute Euclidean distance between two positions
double euclidean_distance_glowso(double *pos1, double *pos2, int dim) {
    double dist = 0.0;
    for (int j = 0; j < dim; j++) {
        dist += (pos1[j] - pos2[j]) * (pos1[j] - pos2[j]);
    }
    return sqrt(dist);
}

// Select neighbor using roulette wheel selection
int select_by_roulette(double *probs, int size) {
    double cum_prob = 0.0;
    double rn = rand_double(0.0, 1.0);
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
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = convert_to_min(objective_function(opt->population[i].position));
        opt->population[i].fitness = (1.0 - LUCIFERIN_DECAY) * opt->population[i].fitness +
                                     LUCIFERIN_ENHANCEMENT * fitness;
    }
}

// Movement Phase
void movement_phase_glowso(Optimizer *opt, double *decision_range) {
    double *distances = (double *)malloc(opt->population_size * sizeof(double));
    int *neighbors = (int *)malloc(opt->population_size * sizeof(int));
    double *probs = (double *)malloc(opt->population_size * sizeof(double));
    double *current_pos = (double *)malloc(opt->dim * sizeof(double));

    for (int i = 0; i < opt->population_size; i++) {
        // Copy current position
        for (int j = 0; j < opt->dim; j++) {
            current_pos[j] = opt->population[i].position[j];
        }
        double current_luciferin = opt->population[i].fitness;

        // Compute distances and find neighbors
        int neighbor_count = 0;
        for (int j = 0; j < opt->population_size; j++) {
            distances[j] = euclidean_distance_glowso(current_pos, opt->population[j].position, opt->dim);
            if (distances[j] < decision_range[i] && opt->population[j].fitness > current_luciferin) {
                neighbors[neighbor_count] = j;
                neighbor_count++;
            }
        }

        if (neighbor_count == 0) {
            // No movement if no neighbors
            continue;
        }

        // Compute probabilities for roulette wheel selection
        double prob_sum = 0.0;
        for (int k = 0; k < neighbor_count; k++) {
            probs[k] = opt->population[neighbors[k]].fitness - current_luciferin;
            prob_sum += probs[k];
        }
        for (int k = 0; k < neighbor_count; k++) {
            probs[k] /= prob_sum;
        }

        // Select a neighbor
        int selected_idx = neighbors[select_by_roulette(probs, neighbor_count)];
        double *selected_pos = opt->population[selected_idx].position;
        double distance = euclidean_distance_glowso(selected_pos, current_pos, opt->dim);

        if (distance > 0) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = current_pos[j] +
                                                GSO_STEP_SIZE * (selected_pos[j] - current_pos[j]) / distance;
            }
        }
    }

    free(distances);
    free(neighbors);
    free(probs);
    free(current_pos);
    enforce_bound_constraints(opt);
}

// Decision Range Update Phase
void decision_range_update(Optimizer *opt, double *decision_range) {
    double *distances = (double *)malloc(opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        double current_luciferin = opt->population[i].fitness;
        int neighbor_count = 0;
        for (int j = 0; j < opt->population_size; j++) {
            distances[j] = euclidean_distance_glowso(opt->population[i].position, opt->population[j].position, opt->dim);
            if (distances[j] < decision_range[i] && opt->population[j].fitness > current_luciferin) {
                neighbor_count++;
            }
        }
        decision_range[i] = fmin(SENSOR_RANGE,
                                 fmax(0.0, decision_range[i] +
                                           NEIGHBOR_THRESHOLD * (NEIGHBOR_COUNT - neighbor_count)));
    }
    free(distances);
}

// Main Optimization Function
void GlowSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Allocate decision range array
    double *decision_range = (double *)malloc(opt->population_size * sizeof(double));
    if (!decision_range) {
        fprintf(stderr, "Failed to allocate decision_range\n");
        return;
    }

    // Initialize luciferin and decision range
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = LUCIFERIN_INITIAL;
        decision_range[i] = DECISION_RANGE_INITIAL;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        luciferin_update(opt, objective_function);

        // Update best solution
        int best_idx = 0;
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].fitness > opt->population[best_idx].fitness) {
                best_idx = i;
            }
        }
        double fitness = convert_to_min(objective_function(opt->population[best_idx].position));
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[best_idx].position[j];
            }
        }

        movement_phase_glowso(opt, decision_range);
        decision_range_update(opt, decision_range);

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(decision_range);
}
