#include "IWD.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Utility function
double rand_double(double min, double max);

// Compute g(soil) for probability calculation
static inline double g_soil(double *restrict soil, int population_size, int i, int j, 
                            char *restrict visited_flags) {
    double minimum = INFINITY;
    for (int l = 0; l < population_size; l++) {
        if (!visited_flags[l]) {
            if (soil[i * population_size + l] < minimum) {
                minimum = soil[i * population_size + l];
            }
        }
    }
    if (minimum >= 0) {
        return soil[i * population_size + j];
    }
    return soil[i * population_size + j] - minimum;
}

// Compute f(soil) for probability calculation
static inline double f_soil(double *restrict soil, int population_size, int i, int j, 
                            char *restrict visited_flags) {
    return 1.0 / (EPSILON_S + g_soil(soil, population_size, i, j, visited_flags));
}

// Compute probability of choosing j from i
static inline double probability_of_choosing_j(double *restrict soil, int population_size, 
                                              int i, int j, char *restrict visited_flags) {
    double sum_fsoil = 0.0;
    for (int k = 0; k < population_size; k++) {
        if (!visited_flags[k]) {
            sum_fsoil += f_soil(soil, population_size, i, k, visited_flags);
        }
    }
    return sum_fsoil > 0.0 ? f_soil(soil, population_size, i, j, visited_flags) / sum_fsoil : 0.0;
}

// Initialize IWD population
void initialize_iwd_population(Optimizer *restrict opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Move water drop
void move_water_drop(Optimizer *restrict opt, int iwd_idx, int *restrict visited, int *restrict visited_count, 
                     double *restrict soil_amount, double *restrict soil, double *restrict hud, 
                     char *restrict visited_flags, int *restrict valid_nodes, double *restrict probabilities) {
    int current = iwd_idx;
    double velocity = INIT_VEL;
    int valid_count = 0;

    // Collect valid (unvisited) nodes
    for (int j = 0; j < opt->population_size; j++) {
        if (!visited_flags[j]) {
            valid_nodes[valid_count] = j;
            probabilities[valid_count] = probability_of_choosing_j(soil, opt->population_size, current, j, visited_flags);
            valid_count++;
        }
    }

    // Select next node
    double random_number = rand_double(0.0, 1.0);
    double probability_sum = 0.0;
    int next_node = current;
    int node_selected = 0;

    for (int v = 0; v < valid_count; v++) {
        probability_sum += probabilities[v];
        if (random_number < probability_sum) {
            next_node = valid_nodes[v];
            node_selected = 1;
            break;
        }
    }

    if (node_selected) {
        // Update velocity
        velocity += A_V / (B_V + C_V * soil[current * opt->population_size + next_node] * 
                          soil[current * opt->population_size + next_node]);
        
        // Update soil
        double delta_soil = A_S / (B_S + C_S * hud[current * opt->population_size + next_node] / velocity * 
                                  hud[current * opt->population_size + next_node] / velocity);
        soil[current * opt->population_size + next_node] = (1 - P_N) * soil[current * opt->population_size + next_node] - 
                                                          P_N * delta_soil;
        *soil_amount += delta_soil;

        // Move water drop (interpolate position)
        for (int k = 0; k < opt->dim; k++) {
            opt->population[iwd_idx].position[k] = 0.5 * (opt->population[current].position[k] + 
                                                        opt->population[next_node].position[k]);
        }
        visited[*visited_count] = next_node;
        visited_flags[next_node] = 1;
        (*visited_count)++;
    }

    enforce_bound_constraints(opt);
}

// Update iteration best
void update_iteration_best(Optimizer *restrict opt, int *restrict visited, int visited_count, 
                          double soil_amount, int iwd_idx, double *restrict soil) {
    for (int i = 0; i < visited_count - 1; i++) {
        int prev = visited[i];
        int curr = visited[i + 1];
        soil[prev * opt->population_size + curr] = (1 + P_IWD) * soil[prev * opt->population_size + curr] -
                                                  P_IWD * (1.0 / (opt->population_size - 1)) * soil_amount;
    }
}

// Main optimization function
void IWD_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict)) {
    // Allocate memory
    int *visited = (int *)malloc(opt->population_size * sizeof(int));
    int *visited_counts = (int *)malloc(opt->population_size * sizeof(int));
    double *qualities = (double *)malloc(opt->population_size * sizeof(double));
    double *soil_amounts = (double *)malloc(opt->population_size * sizeof(double));
    double *soil = (double *)malloc(opt->population_size * opt->population_size * sizeof(double));
    double *hud = (double *)malloc(opt->population_size * opt->population_size * sizeof(double));
    char *visited_flags = (char *)malloc(opt->population_size * sizeof(char));
    int *valid_nodes = (int *)malloc(opt->population_size * sizeof(int));
    double *probabilities = (double *)malloc(opt->population_size * sizeof(double));

    // Initialize soil and HUD matrices
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->population_size; j++) {
            soil[i * opt->population_size + j] = INITIAL_SOIL;
            if (i != j) {
                double dist = 0.0;
                for (int k = 0; k < opt->dim; k++) {
                    double diff = opt->population[i].position[k] - opt->population[j].position[k];
                    dist += diff * diff;
                }
                hud[i * opt->population_size + j] = sqrt(dist);
            } else {
                hud[i * opt->population_size + j] = 0.0;
            }
        }
    }

    initialize_iwd_population(opt);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        int max_quality_idx = 0;
        double max_quality = -INFINITY;
        int best_visited_count = 0;

        for (int i = 0; i < opt->population_size; i++) {
            visited_counts[i] = 1;
            visited[0] = i;
            soil_amounts[i] = 0.0;
            for (int j = 0; j < opt->population_size; j++) {
                visited_flags[j] = (j == i);
            }

            // Move water drop until all nodes are visited or no valid move
            while (visited_counts[i] < opt->population_size) {
                move_water_drop(opt, i, visited, &visited_counts[i], &soil_amounts[i], soil, hud, 
                                visited_flags, valid_nodes, probabilities);
            }

            // Complete cycle by returning to start
            int start = i;
            if (!visited_flags[start]) {
                double velocity = INIT_VEL + A_V / (B_V + C_V * soil[i * opt->population_size + start] * 
                                                   soil[i * opt->population_size + start]);
                double delta_soil = A_S / (B_S + C_S * hud[i * opt->population_size + start] / velocity * 
                                          hud[i * opt->population_size + start] / velocity);
                soil[i * opt->population_size + start] = (1 - P_N) * soil[i * opt->population_size + start] - 
                                                        P_N * delta_soil;
                soil_amounts[i] += delta_soil;
                visited[visited_counts[i]] = start;
                visited_counts[i]++;
                visited_flags[start] = 1;
            }

            // Compute quality
            double fitness = objective_function(opt->population[i].position);
            qualities[i] = (fitness != 0.0) ? 1.0 / fitness : INFINITY;

            // Track best quality and corresponding visited count
            if (qualities[i] > max_quality) {
                max_quality = qualities[i];
                max_quality_idx = i;
                best_visited_count = visited_counts[i];
            }
        }

        // Update iteration best with the best water drop's visited list
        update_iteration_best(opt, visited, best_visited_count, soil_amounts[max_quality_idx], max_quality_idx, soil);

        // Update global best
        double current_fitness = objective_function(opt->population[max_quality_idx].position);
        if (current_fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = current_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[max_quality_idx].position[j];
            }
        }

        // Reinitialize HUD for new positions
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = i + 1; j < opt->population_size; j++) {
                double dist = 0.0;
                for (int k = 0; k < opt->dim; k++) {
                    double diff = opt->population[i].position[k] - opt->population[j].position[k];
                    dist += diff * diff;
                }
                dist = sqrt(dist);
                hud[i * opt->population_size + j] = dist;
                hud[j * opt->population_size + i] = dist;
            }
            hud[i * opt->population_size + i] = 0.0;
        }
    }

    // Free memory
    free(visited);
    free(visited_counts);
    free(qualities);
    free(soil_amounts);
    free(soil);
    free(hud);
    free(visited_flags);
    free(valid_nodes);
    free(probabilities);
}
