#include "TS.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define NEIGHBORHOOD_SIZE 20
#define TABU_TENURE 10

static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static inline void clamp_position(double *position, const double *bounds, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (position[i] < bounds[2 * i]) position[i] = bounds[2 * i];
        if (position[i] > bounds[2 * i + 1]) position[i] = bounds[2 * i + 1];
    }
}

void TS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    const int dim = opt->dim;
    const int max_iter = opt->max_iter;
    const int pop_size = opt->population_size;

    double *current_solution = aligned_alloc(64, sizeof(double) * dim);
    double *best_solution = aligned_alloc(64, sizeof(double) * dim);
    double *neighbor = aligned_alloc(64, sizeof(double) * dim);
    double *move = aligned_alloc(64, sizeof(double) * dim);
    int *tabu_list = calloc(dim * 2, sizeof(int));  // allow signed directions

    // Initial solution
    for (int i = 0; i < dim; ++i) {
        current_solution[i] = rand_double(opt->bounds[2 * i], opt->bounds[2 * i + 1]);
        best_solution[i] = current_solution[i];
    }
    double current_value = objective_function(current_solution);
    double best_value = current_value;

    for (int iter = 0; iter < max_iter; ++iter) {
        double best_candidate_value = DBL_MAX;
        int move_index = -1;

        for (int n = 0; n < NEIGHBORHOOD_SIZE; ++n) {
            for (int i = 0; i < dim; ++i) {
                double delta = rand_double(-0.1, 0.1);
                neighbor[i] = current_solution[i] + delta;
                move[i] = (delta > 0.0) ? 1 : -1;
            }

            clamp_position(neighbor, opt->bounds, dim);

            // Hash move to an index
            int hash = 0;
            for (int i = 0; i < dim; ++i) {
                hash += (move[i] > 0 ? i + 1 : -(i + 1)) * 31;
            }
            hash = abs(hash) % (dim * 2);

            if (tabu_list[hash] > 0)
                continue;

            double candidate_value = objective_function(neighbor);
            if (candidate_value < best_candidate_value) {
                best_candidate_value = candidate_value;
                memcpy(current_solution, neighbor, sizeof(double) * dim);
                move_index = hash;
            }
        }

        if (move_index >= 0) {
            tabu_list[move_index] = TABU_TENURE;
        }

        if (current_value < best_value) {
            best_value = current_value;
            memcpy(best_solution, current_solution, sizeof(double) * dim);
        }

        for (int i = 0; i < dim * 2; ++i) {
            if (tabu_list[i] > 0)
                tabu_list[i]--;
        }

        current_value = objective_function(current_solution);
    }

    memcpy(opt->best_solution.position, best_solution, sizeof(double) * dim);
    opt->best_solution.fitness = best_value;

    free(current_solution);
    free(best_solution);
    free(neighbor);
    free(move);
    free(tabu_list);
}

