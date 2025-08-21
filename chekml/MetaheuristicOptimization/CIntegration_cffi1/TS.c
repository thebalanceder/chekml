#include <immintrin.h>  // For AVX2 intrinsics
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "TS.h"

#define ALIGNMENT 32
#define MAX_TABU_MOVES 256  // circular buffer

// Preallocated aligned memory buffers
static double *aligned_alloc_double(size_t n) {
    return (double *)aligned_alloc(ALIGNMENT, n * sizeof(double));
}

static inline void clamp_to_bounds(double *restrict x, const double *restrict lb, const double *restrict ub, int dim) {
    for (int i = 0; i < dim; i++) {
        if (x[i] < lb[i]) x[i] = lb[i];
        else if (x[i] > ub[i]) x[i] = ub[i];
    }
}

static inline void add_tabu_move(double *tabu_moves, int *tabu_index, double *move, int dim) {
    memcpy(&tabu_moves[*tabu_index * dim], move, sizeof(double) * dim);
    *tabu_index = (*tabu_index + 1) % MAX_TABU_MOVES;
}

static inline int is_tabu_move(double *tabu_moves, int tabu_index, double *move, int dim) {
    for (int i = 0; i < MAX_TABU_MOVES; i++) {
        double *m = &tabu_moves[i * dim];
        int is_same = 1;
        for (int j = 0; j < dim; j++) {
            if (fabs(m[j] - move[j]) > 1e-4) {
                is_same = 0;
                break;
            }
        }
        if (is_same) return 1;
    }
    return 0;
}

void TS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int iter, i, j;

    double *current = aligned_alloc_double(dim);
    double *best = aligned_alloc_double(dim);
    double *neighbor = aligned_alloc_double(dim);
    double *move = aligned_alloc_double(dim);
    double *tabu_moves = aligned_alloc_double(MAX_TABU_MOVES * dim);
    int tabu_index = 0;

    // Initial solution
    for (i = 0; i < dim; i++) {
        current[i] = opt->bounds[2 * i] + ((double)rand() / RAND_MAX) * (opt->bounds[2 * i + 1] - opt->bounds[2 * i]);
        best[i] = current[i];
    }

    double best_value = objective_function(best);

    for (iter = 0; iter < opt->max_iter; iter++) {
        double best_candidate_value = DBL_MAX;
        memcpy(neighbor, current, sizeof(double) * dim);

        for (int k = 0; k < opt->population_size; k++) {
            // Generate neighbor
            for (j = 0; j < dim; j++) {
                double r = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
                neighbor[j] = current[j] + r;
                move[j] = r;
            }

            clamp_to_bounds(neighbor, opt->bounds, opt->bounds + 1, dim);

            if (is_tabu_move(tabu_moves, tabu_index, move, dim)) {
                continue;
            }

            double value = objective_function(neighbor);

            if (value < best_candidate_value) {
                best_candidate_value = value;
                memcpy(current, neighbor, sizeof(double) * dim);
                add_tabu_move(tabu_moves, &tabu_index, move, dim);
            }
        }

        if (best_candidate_value < best_value) {
            best_value = best_candidate_value;
            memcpy(best, current, sizeof(double) * dim);
        }
    }

    // Final best solution
    opt->best_solution.fitness = best_value;
    for (i = 0; i < dim; i++) {
        opt->best_solution.position[i] = best[i];
    }

    free(current);
    free(best);
    free(neighbor);
    free(move);
    free(tabu_moves);
}

