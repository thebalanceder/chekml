#include "IWD.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize IWD population
void initialize_iwd_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute g(soil) for probability calculation
double g_soil(double *soil, int population_size, int i, int j, int *visited, int visited_count) {
    double minimum = INFINITY;
    for (int l = 0; l < population_size; l++) {
        int is_visited = 0;
        for (int v = 0; v < visited_count; v++) {
            if (visited[v] == l) {
                is_visited = 1;
                break;
            }
        }
        if (!is_visited) {
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
double f_soil(double *soil, int population_size, int i, int j, int *visited, int visited_count) {
    return 1.0 / (EPSILON_S + g_soil(soil, population_size, i, j, visited, visited_count));
}

// Compute probability of choosing j from i
double probability_of_choosing_j(double *soil, int population_size, int i, int j, int *visited, int visited_count) {
    double sum_fsoil = 0.0;
    for (int k = 0; k < population_size; k++) {
        int is_visited = 0;
        for (int v = 0; v < visited_count; v++) {
            if (visited[v] == k) {
                is_visited = 1;
                break;
            }
        }
        if (!is_visited) {
            sum_fsoil += f_soil(soil, population_size, i, k, visited, visited_count);
        }
    }
    if (sum_fsoil == 0.0) {
        return 0.0;
    }
    return f_soil(soil, population_size, i, j, visited, visited_count) / sum_fsoil;
}

// Compute time to travel from i to j
double time_iwd(double *hud, int population_size, int i, int j, double vel) {
    return hud[i * population_size + j] / vel;
}

// Update velocity
double update_velocity(double *soil, int population_size, double current_vel, int i, int j) {
    return current_vel + A_V / (B_V + C_V * pow(soil[i * population_size + j], 2));
}

// Update soil
double update_soil(double *soil, int population_size, int i, int j, double vel, double *soil_amount) {
    double delta_soil = A_S / (B_S + C_S * pow(time_iwd(soil, population_size, i, j, vel), 2));
    soil[i * population_size + j] = (1 - P_N) * soil[i * population_size + j] - P_N * delta_soil;
    *soil_amount += delta_soil;
    return delta_soil;
}

// Move water drop
void move_water_drop(Optimizer *opt, int iwd_idx, int *visited, int *visited_count, double *soil_amount, double *soil, double *hud) {
    int current = iwd_idx;
    double velocity = INIT_VEL;
    double *probabilities = (double *)malloc(opt->population_size * sizeof(double));
    int *valid_nodes = (int *)malloc(opt->population_size * sizeof(int));
    int valid_count = 0;

    // Collect valid (unvisited) nodes
    for (int j = 0; j < opt->population_size; j++) {
        int is_visited = 0;
        for (int v = 0; v < *visited_count; v++) {
            if (visited[v] == j) {
                is_visited = 1;
                break;
            }
        }
        if (!is_visited) {
            valid_nodes[valid_count] = j;
            probabilities[valid_count] = probability_of_choosing_j(soil, opt->population_size, current, j, visited, *visited_count);
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
        // Update velocity and soil
        velocity = update_velocity(soil, opt->population_size, velocity, current, next_node);
        update_soil(soil, opt->population_size, current, next_node, velocity, soil_amount);
        // Move water drop (interpolate position)
        for (int k = 0; k < opt->dim; k++) {
            opt->population[iwd_idx].position[k] = (opt->population[current].position[k] + opt->population[next_node].position[k]) / 2;
        }
        visited[*visited_count] = next_node;
        (*visited_count)++;
    }

    free(probabilities);
    free(valid_nodes);
    enforce_bound_constraints(opt);
}

// Update iteration best
void update_iteration_best(Optimizer *opt, int *visited, int visited_count, double soil_amount, int iwd_idx, double *soil) {
    // Update soil for iteration best path
    for (int i = 0; i < visited_count - 1; i++) {
        int prev = visited[i];
        int curr = visited[i + 1];
        soil[prev * opt->population_size + curr] = (1 + P_IWD) * soil[prev * opt->population_size + curr] -
                                                  P_IWD * (1.0 / (opt->population_size - 1)) * soil_amount;
    }
}

// Main optimization function
void IWD_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    int *visited = (int *)malloc(opt->population_size * sizeof(int));
    int *visited_counts = (int *)malloc(opt->population_size * sizeof(int));
    double *qualities = (double *)malloc(opt->population_size * sizeof(double));
    double *soil_amounts = (double *)malloc(opt->population_size * sizeof(double));
    double *soil = (double *)malloc(opt->population_size * opt->population_size * sizeof(double));
    double *hud = (double *)malloc(opt->population_size * opt->population_size * sizeof(double));

    // Initialize soil and HUD matrices
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->population_size; j++) {
            soil[i * opt->population_size + j] = INITIAL_SOIL;
            if (i != j) {
                double dist = 0.0;
                for (int k = 0; k < opt->dim; k++) {
                    dist += pow(opt->population[i].position[k] - opt->population[j].position[k], 2);
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

            // Move water drop until all nodes are visited or no valid move
            while (visited_counts[i] < opt->population_size) {
                move_water_drop(opt, i, visited, &visited_counts[i], &soil_amounts[i], soil, hud);
            }

            // Complete cycle by returning to start
            int start = i;
            int start_visited = 0;
            for (int v = 0; v < visited_counts[i]; v++) {
                if (visited[v] == start) {
                    start_visited = 1;
                    break;
                }
            }
            if (!start_visited) {
                double velocity = update_velocity(soil, opt->population_size, INIT_VEL, i, start);
                update_soil(soil, opt->population_size, i, start, velocity, &soil_amounts[i]);
                visited[visited_counts[i]] = start;
                visited_counts[i]++;
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
            for (int j = 0; j < opt->population_size; j++) {
                if (i != j) {
                    double dist = 0.0;
                    for (int k = 0; k < opt->dim; k++) {
                        dist += pow(opt->population[i].position[k] - opt->population[j].position[k], 2);
                    }
                    hud[i * opt->population_size + j] = sqrt(dist);
                } else {
                    hud[i * opt->population_size + j] = 0.0;
                }
            }
        }
    }

    free(visited);
    free(visited_counts);
    free(qualities);
    free(soil_amounts);
    free(soil);
    free(hud);
}
