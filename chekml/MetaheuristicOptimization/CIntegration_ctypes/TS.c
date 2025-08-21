#include "TS.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void init_tabu_list(TabuList *tabu, int capacity, int dim) {
    tabu->capacity = capacity;
    tabu->dim = dim;
    tabu->count = 0;
    tabu->moves = (double *)malloc(sizeof(double) * capacity * dim);
    tabu->tenures = (int *)malloc(sizeof(int) * capacity);
}

void free_tabu_list(TabuList *tabu) {
    free(tabu->moves);
    free(tabu->tenures);
}

int is_move_tabu(TabuList *tabu, double *move) {
    for (int i = 0; i < tabu->count; i++) {
        int offset = i * tabu->dim;
        int same = 1;
        for (int d = 0; d < tabu->dim; d++) {
            if (round(tabu->moves[offset + d] * 10000.0) != round(move[d] * 10000.0)) {
                same = 0;
                break;
            }
        }
        if (same) return 1;
    }
    return 0;
}

void add_tabu_move(TabuList *tabu, double *move) {
    if (tabu->count < tabu->capacity) {
        int offset = tabu->count * tabu->dim;
        memcpy(&tabu->moves[offset], move, sizeof(double) * tabu->dim);
        tabu->tenures[tabu->count] = TABU_TENURE;
        tabu->count++;
    }
}

void decrement_tabu_tenures(TabuList *tabu) {
    int i = 0;
    while (i < tabu->count) {
        tabu->tenures[i]--;
        if (tabu->tenures[i] <= 0) {
            // Shift everything left
            for (int j = i; j < tabu->count - 1; j++) {
                memcpy(&tabu->moves[j * tabu->dim], &tabu->moves[(j + 1) * tabu->dim], sizeof(double) * tabu->dim);
                tabu->tenures[j] = tabu->tenures[j + 1];
            }
            tabu->count--;
        } else {
            i++;
        }
    }
}

void generate_neighbor(double *neighbor, double *current, double *lower, double *upper, int dim) {
    for (int i = 0; i < dim; i++) {
        double step = ((double)rand() / RAND_MAX) * 2.0 * STEP_SIZE - STEP_SIZE;
        neighbor[i] = current[i] + step;
        if (neighbor[i] < lower[i]) neighbor[i] = lower[i];
        if (neighbor[i] > upper[i]) neighbor[i] = upper[i];
    }
}

void TS_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double *current = (double *)malloc(sizeof(double) * opt->dim);
    double *best = (double *)malloc(sizeof(double) * opt->dim);
    double *neighbor = (double *)malloc(sizeof(double) * opt->dim);
    double *move = (double *)malloc(sizeof(double) * opt->dim);

    double *lower = (double *)malloc(sizeof(double) * opt->dim);
    double *upper = (double *)malloc(sizeof(double) * opt->dim);
    for (int i = 0; i < opt->dim; i++) {
        lower[i] = opt->bounds[2 * i];
        upper[i] = opt->bounds[2 * i + 1];
    }

    for (int i = 0; i < opt->dim; i++) {
        current[i] = lower[i] + ((double)rand() / RAND_MAX) * (upper[i] - lower[i]);
        best[i] = current[i];
    }

    double current_val = objective_function(current);
    double best_val = current_val;

    TabuList tabu;
    init_tabu_list(&tabu, NEIGHBORHOOD_SIZE * 2, opt->dim);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        double best_candidate_val = INFINITY;
        double best_candidate[opt->dim];
        double best_move[opt->dim];

        for (int n = 0; n < NEIGHBORHOOD_SIZE; n++) {
            generate_neighbor(neighbor, current, lower, upper, opt->dim);
            for (int d = 0; d < opt->dim; d++) {
                move[d] = neighbor[d] - current[d];
            }

            if (is_move_tabu(&tabu, move)) continue;

            double val = objective_function(neighbor);
            if (val < best_candidate_val) {
                memcpy(best_candidate, neighbor, sizeof(double) * opt->dim);
                memcpy(best_move, move, sizeof(double) * opt->dim);
                best_candidate_val = val;
            }
        }
		/*this is for aspiration criteria 
		for (int n = 0; n < NEIGHBORHOOD_SIZE; n++) {
			generate_neighbor(neighbor, current, lower, upper, opt->dim);
			for (int d = 0; d < opt->dim; d++) {
				move[d] = neighbor[d] - current[d];
			}

			double val = objective_function(neighbor);

			// Aspiration criterion: allow tabu move if it improves best_val
			if (is_move_tabu(&tabu, move) && val >= best_val) continue;

			if (val < best_candidate_val) {
				memcpy(best_candidate, neighbor, sizeof(double) * opt->dim);
				memcpy(best_move, move, sizeof(double) * opt->dim);
				best_candidate_val = val;
			}
		}
		*/

        if (best_candidate_val < INFINITY) {
            memcpy(current, best_candidate, sizeof(double) * opt->dim);
            current_val = best_candidate_val;

            if (current_val < best_val) {
                memcpy(best, current, sizeof(double) * opt->dim);
                best_val = current_val;
            }

            add_tabu_move(&tabu, best_move);
        }

        decrement_tabu_tenures(&tabu);
    }

    // Store best result back into optimizer
    opt->best_solution.fitness = best_val;
    for (int i = 0; i < opt->dim; i++) {
        opt->best_solution.position[i] = best[i];
    }

    free(current);
    free(best);
    free(neighbor);
    free(move);
    free(lower);
    free(upper);
    free_tabu_list(&tabu);
}

