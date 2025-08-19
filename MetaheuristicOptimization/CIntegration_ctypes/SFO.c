#include "SFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>  // For memcpy

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Calculate Instruction(t) as per Eq. (1)
double calculate_instruction(int t, int max_iter) {
    return (1.0 - 0.15 * rand_double(0.0, 1.0)) * (1.0 - (double)t / max_iter);
}

// Calculate probability of losing contact p(t) as per Eq. (3)
double calculate_loss_probability(int t, int max_iter) {
    return P0 * cos(M_PI * t / (2.0 * max_iter));
}

// Calculate raid coefficient w(t) as per Eq. (7)
double calculate_raid_coefficient(int t, int max_iter) {
    return 0.75 - 0.55 * atan(pow((double)t / max_iter, 2.0 * M_PI));
}

// Large-Scale Search as per Eq. (4)
void large_scale_search(Optimizer *opt, int idx, double r1) {
    if (r1 >= 0.5) {
        double *new_pos = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            double term1 = r1 * (opt->best_solution.position[j] - opt->population[idx].position[j]);
            double term2 = (1.0 - r1) * (ub - lb) * (rand_double(0.0, 1.0) < 0.5 ? -1.0 : 1.0);
            new_pos[j] = opt->population[idx].position[j] + term1 + term2;
        }
        memcpy(opt->population[idx].position, new_pos, opt->dim * sizeof(double));
        free(new_pos);
        enforce_bound_constraints(opt);
    }
}

// Raid Strategy as per Eq. (5, 6)
void raid(Optimizer *opt, int idx, double r1, double w) {
    if (r1 < 0.5) {
        // Select random aim index
        int available_indices[opt->population_size - 1];
        int count = 0;
        for (int i = 0; i < opt->population_size; i++) {
            if (i != idx) available_indices[count++] = i;
        }
        if (count == 0) return;
        int aim_idx = available_indices[rand() % count];

        double f_i = opt->population[idx].fitness;
        double f_aim = opt->population[aim_idx].fitness;
        double *A_i = (double *)malloc(opt->dim * sizeof(double));

        // Calculate search vector A_i as per Eq. (6)
        if (f_i + f_aim != 0.0) {
            for (int j = 0; j < opt->dim; j++) {
                A_i[j] = (f_i / (f_i + f_aim)) * (opt->population[aim_idx].position[j] - opt->population[idx].position[j]);
            }
        } else {
            for (int j = 0; j < opt->dim; j++) {
                A_i[j] = 0.0;
            }
        }

        // Update position as per Eq. (5)
        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] += w * A_i[j];
        }
        free(A_i);
        enforce_bound_constraints(opt);
    }
}

// Transition Phase as per Eq. (8)
void transition_phase(Optimizer *opt, int idx, double r2, double w, double instruction) {
    double *new_pos = (double *)malloc(opt->dim * sizeof(double));
    if (r2 >= 0.5) {
        // Reuse raid strategy with r1 < 0.5
        raid(opt, idx, 0.4, w);
    } else {
        for (int j = 0; j < opt->dim; j++) {
            new_pos[j] = instruction * (opt->best_solution.position[j] - opt->population[idx].position[j]) + 0.1 * opt->population[idx].position[j];
        }
        memcpy(opt->population[idx].position, new_pos, opt->dim * sizeof(double));
        enforce_bound_constraints(opt);
    }
    free(new_pos);
}

// Arrest-Rescue Strategy as per Eq. (9, 10)
void arrest_rescue(Optimizer *opt) {
    double *X_ave = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        X_ave[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            X_ave[j] += opt->population[i].position[j];
        }
        X_ave[j] /= opt->population_size;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double r = rand_double(-1.0, 1.0);
            opt->population[i].position[j] = opt->best_solution.position[j] + r * fabs(opt->best_solution.position[j] - X_ave[j]);
        }
    }
    free(X_ave);
    enforce_bound_constraints(opt);
}

// Unmanned Search as per Eq. (11, 12)
void unmanned_search(Optimizer *opt, int t) {
    double *v = (double *)malloc(opt->dim * sizeof(double));
    double *c = (double *)malloc(opt->dim * sizeof(double));
    double sum_squares = 0.0;

    // Calculate c as per Eq. (12)
    for (int j = 0; j < opt->dim; j++) {
        double lb = opt->bounds[2 * j];
        double ub = opt->bounds[2 * j + 1];
        c[j] = K * (lb + (1.0 - (double)t / opt->max_iter) * (ub - lb));
    }

    // Generate random vector v
    for (int j = 0; j < opt->dim; j++) {
        v[j] = rand_double(-1.0, 1.0);
        sum_squares += v[j] * v[j];
    }

    // Normalize v to magnitude c
    double norm = sqrt(sum_squares);
    if (norm != 0.0) {
        for (int j = 0; j < opt->dim; j++) {
            v[j] = v[j] / norm * c[j];
        }
    }

    // Select random base position and update
    int base_idx = rand() % opt->population_size;
    for (int j = 0; j < opt->dim; j++) {
        opt->population[base_idx].position[j] += v[j];
    }

    free(v);
    free(c);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void SFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Evaluate initial population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        // Unmanned search
        unmanned_search(opt, t);
        opt->population[0].fitness = objective_function(opt->population[0].position);
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
        }

        // Calculate iteration parameters
        double instruction = calculate_instruction(t, opt->max_iter);
        double p = calculate_loss_probability(t, opt->max_iter);
        double w = calculate_raid_coefficient(t, opt->max_iter);

        // Update population
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double(0.0, 1.0) < p) {
                continue; // Simulate loss of contact
            }

            double r1 = rand_double(0.0, 1.0);
            double r2 = rand_double(0.0, 1.0);

            if (instruction >= TV1) { // Exploration phase
                large_scale_search(opt, i, r1);
                if (opt->population[i].fitness == INFINITY) { // If large_scale_search didn't update
                    raid(opt, i, r1, w);
                }
            } else if (instruction > TV2 && instruction < TV1) { // Transition phase
                transition_phase(opt, i, r2, w, instruction);
            } else { // Exploitation phase
                arrest_rescue(opt);
            }

            // Update fitness
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        enforce_bound_constraints(opt);
        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }
}
