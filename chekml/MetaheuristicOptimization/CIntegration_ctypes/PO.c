#include "PO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Election Phase
void election_phase(Optimizer *opt, double *fitness, double *a_winners, int *a_winner_indices) {
    int areas = opt->population_size / PARTIES;
    for (int a = 0; a < areas; a++) {
        int start_idx = a;
        int end_idx = opt->population_size;
        int step = areas;
        double min_fitness = INFINITY;
        int min_idx = -1;

        // Find the best solution in the constituency
        for (int i = start_idx; i < end_idx; i += step) {
            if (fitness[i] < min_fitness) {
                min_fitness = fitness[i];
                min_idx = i;
            }
        }

        a_winner_indices[a] = min_idx;
        for (int j = 0; j < opt->dim; j++) {
            a_winners[a * opt->dim + j] = opt->population[min_idx].position[j];
        }
    }
}

// Government Formation Phase
void government_formation_phase(Optimizer *opt, double *fitness, ObjectiveFunction objective_function) {
    int areas = opt->population_size / PARTIES;
    for (int p = 0; p < PARTIES; p++) {
        int party_start = p * areas;
        double min_fitness = INFINITY;
        int party_leader_idx = -1;

        // Find party leader
        for (int a = 0; a < areas; a++) {
            int idx = party_start + a;
            if (fitness[idx] < min_fitness) {
                min_fitness = fitness[idx];
                party_leader_idx = idx;
            }
        }

        // Update positions towards party leader
        for (int a = 0; a < areas; a++) {
            int member_idx = party_start + a;
            if (member_idx != party_leader_idx) {
                double *new_pos = (double *)malloc(opt->dim * sizeof(double));
                double r = rand_double(0.0, 1.0);
                for (int j = 0; j < opt->dim; j++) {
                    new_pos[j] = opt->population[member_idx].position[j] +
                                 r * (opt->population[party_leader_idx].position[j] - opt->population[member_idx].position[j]);
                    new_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos[j]));
                }
                double new_fitness = objective_function(new_pos);
                if (new_fitness < fitness[member_idx]) {
                    for (int j = 0; j < opt->dim; j++) {
                        opt->population[member_idx].position[j] = new_pos[j];
                    }
                    fitness[member_idx] = new_fitness;
                }
                free(new_pos);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Election Campaign Phase
void election_campaign_phase(Optimizer *opt, double *fitness, double *prev_positions, ObjectiveFunction objective_function) {
    int areas = opt->population_size / PARTIES;
    for (int p = 0; p < PARTIES; p++) {
        int party_start = p * areas;
        for (int a = 0; a < areas; a++) {
            int member_idx = party_start + a;
            double *new_pos = (double *)malloc(opt->dim * sizeof(double));
            for (int j = 0; j < opt->dim; j++) {
                double r = rand_double(0.0, 1.0);
                new_pos[j] = opt->population[member_idx].position[j] +
                             r * (opt->population[member_idx].position[j] - prev_positions[member_idx * opt->dim + j]);
                new_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_pos[j]));
            }
            double new_fitness = objective_function(new_pos);
            if (new_fitness < fitness[member_idx]) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[member_idx].position[j] = new_pos[j];
                }
                fitness[member_idx] = new_fitness;
            }
            free(new_pos);
        }
    }
    enforce_bound_constraints(opt);
}

// Party Switching Phase
void party_switching_phase(Optimizer *opt, double *fitness, int t) {
    int areas = opt->population_size / PARTIES;
    double psr = (1.0 - t * (1.0 / opt->max_iter)) * LAMBDA_RATE;

    for (int p = 0; p < PARTIES; p++) {
        for (int a = 0; a < areas; a++) {
            int from_idx = p * areas + a;
            if (rand_double(0.0, 1.0) < psr) {
                int to_party = rand() % PARTIES;
                while (to_party == p) {
                    to_party = rand() % PARTIES;
                }
                int to_start = to_party * areas;
                double max_fitness = -INFINITY;
                int to_least_fit_idx = -1;

                // Find least fit member in to_party
                for (int i = to_start; i < to_start + areas; i++) {
                    if (fitness[i] > max_fitness) {
                        max_fitness = fitness[i];
                        to_least_fit_idx = i;
                    }
                }

                // Swap positions
                for (int j = 0; j < opt->dim; j++) {
                    double temp = opt->population[to_least_fit_idx].position[j];
                    opt->population[to_least_fit_idx].position[j] = opt->population[from_idx].position[j];
                    opt->population[from_idx].position[j] = temp;
                }
                // Swap fitness
                double temp_fitness = fitness[to_least_fit_idx];
                fitness[to_least_fit_idx] = fitness[from_idx];
                fitness[from_idx] = temp_fitness;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Parliamentarism Phase
void parliamentarism_phase(Optimizer *opt, double *fitness, double *a_winners, int *a_winner_indices, ObjectiveFunction objective_function) {
    int areas = opt->population_size / PARTIES;
    for (int a = 0; a < areas; a++) {
        double *new_winner = (double *)malloc(opt->dim * sizeof(double));
        memcpy(new_winner, &a_winners[a * opt->dim], opt->dim * sizeof(double));
        int winner_idx = a_winner_indices[a];

        int to_area = rand() % areas;
        while (to_area == a) {
            to_area = rand() % areas;
        }

        for (int j = 0; j < opt->dim; j++) {
            double to_winner_j = a_winners[to_area * opt->dim + j];
            double distance = fabs(to_winner_j - new_winner[j]);
            new_winner[j] = to_winner_j + (2.0 * rand_double(0.0, 1.0) - 1.0) * distance;
            new_winner[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], new_winner[j]));
        }

        double new_fitness = objective_function(new_winner);
        if (new_fitness < fitness[winner_idx]) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[winner_idx].position[j] = new_winner[j];
                a_winners[a * opt->dim + j] = new_winner[j];
            }
            fitness[winner_idx] = new_fitness;
        }
        free(new_winner);
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void PO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    int areas = opt->population_size / PARTIES;

    // Allocate auxiliary arrays
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *aux_fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *prev_fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *aux_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *prev_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *a_winners = (double *)malloc(areas * opt->dim * sizeof(double));
    int *a_winner_indices = (int *)malloc(areas * sizeof(int));

    // Initialize fitness
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = objective_function(opt->population[i].position);
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[i];
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
    memcpy(aux_fitness, fitness, opt->population_size * sizeof(double));
    memcpy(prev_fitness, fitness, opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(&aux_positions[i * opt->dim], opt->population[i].position, opt->dim * sizeof(double));
        memcpy(&prev_positions[i * opt->dim], opt->population[i].position, opt->dim * sizeof(double));
    }

    // Initial phases
    election_phase(opt, fitness, a_winners, a_winner_indices);
    memcpy(aux_fitness, fitness, opt->population_size * sizeof(double));
    memcpy(prev_fitness, fitness, opt->population_size * sizeof(double));
    government_formation_phase(opt, fitness, objective_function);

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        memcpy(prev_fitness, aux_fitness, opt->population_size * sizeof(double));
        memcpy(aux_fitness, fitness, opt->population_size * sizeof(double));
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(&prev_positions[i * opt->dim], &aux_positions[i * opt->dim], opt->dim * sizeof(double));
            memcpy(&aux_positions[i * opt->dim], opt->population[i].position, opt->dim * sizeof(double));
        }

        election_campaign_phase(opt, fitness, prev_positions, objective_function);
        party_switching_phase(opt, fitness, t);
        election_phase(opt, fitness, a_winners, a_winner_indices);
        government_formation_phase(opt, fitness, objective_function);
        parliamentarism_phase(opt, fitness, a_winners, a_winner_indices, objective_function);

        // Update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            fitness[i] = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = opt->best_solution.position[j];
                }
            }
        }
        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    // Free allocated memory
    free(fitness);
    free(aux_fitness);
    free(prev_fitness);
    free(aux_positions);
    free(prev_positions);
    free(a_winners);
    free(a_winner_indices);
}
