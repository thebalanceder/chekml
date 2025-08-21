#include "ALO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize antlion and ant populations
void initialize_populations(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double lb = opt->bounds[2 * j];
            double ub = opt->bounds[2 * j + 1];
            opt->population[i].position[j] = rand_double(lb, ub);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Roulette wheel selection based on fitness weights
int roulette_wheel_selection_alo(double *weights, int size) {
    double accumulation = 0.0;
    double *cumsum = (double *)malloc(size * sizeof(double));
    if (!cumsum) return 0;

    for (int i = 0; i < size; i++) {
        accumulation += weights[i];
        cumsum[i] = accumulation;
    }

    double p = rand_double(0.0, accumulation);
    int selected = 0;
    for (int i = 0; i < size; i++) {
        if (cumsum[i] > p) {
            selected = i;
            break;
        }
    }

    free(cumsum);
    return selected;
}

// Random walk phase around a given antlion
void random_walk_phase(Optimizer *opt, int t, double *antlion_positions, int antlion_size) {
    double I = 1.0;
    double T = (double)opt->max_iter;

    // Adjust I based on iteration
    if (t > T / 10) I = 1.0 + I_FACTOR_1 * (t / T);
    if (t > T / 2) I = 1.0 + I_FACTOR_2 * (t / T);
    if (t > T * 3 / 4) I = 1.0 + I_FACTOR_3 * (t / T);
    if (t > T * 0.9) I = 1.0 + I_FACTOR_4 * (t / T);
    if (t > T * 0.95) I = 1.0 + I_FACTOR_5 * (t / T);

    // Temporary arrays for random walks
    double *walks = (double *)malloc((opt->max_iter + 1) * opt->dim * sizeof(double));
    if (!walks) return;

    for (int i = 0; i < antlion_size; i++) {
        double *lb = (double *)malloc(opt->dim * sizeof(double));
        double *ub = (double *)malloc(opt->dim * sizeof(double));
        if (!lb || !ub) {
            free(lb); free(ub); free(walks);
            return;
        }

        // Copy bounds and adjust by I
        for (int j = 0; j < opt->dim; j++) {
            lb[j] = opt->bounds[2 * j] / I;
            ub[j] = opt->bounds[2 * j + 1] / I;
        }

        // Move interval around antlion
        double r = rand_double(0.0, 1.0);
        if (r < 0.5) {
            for (int j = 0; j < opt->dim; j++) lb[j] += antlion_positions[i * opt->dim + j];
        } else {
            for (int j = 0; j < opt->dim; j++) lb[j] = -lb[j] + antlion_positions[i * opt->dim + j];
        }

        r = rand_double(0.0, 1.0);
        if (r >= 0.5) {
            for (int j = 0; j < opt->dim; j++) ub[j] += antlion_positions[i * opt->dim + j];
        } else {
            for (int j = 0; j < opt->dim; j++) ub[j] = -ub[j] + antlion_positions[i * opt->dim + j];
        }

        // Generate random walks for each dimension
        for (int j = 0; j < opt->dim; j++) {
            double *X = (double *)calloc(opt->max_iter + 1, sizeof(double));
            if (!X) {
                free(lb); free(ub); free(walks);
                return;
            }

            for (int k = 0; k < opt->max_iter; k++) {
                r = rand_double(0.0, 1.0);
                X[k + 1] = X[k] + (r > 0.5 ? 1.0 : -1.0);
            }

            double a = X[0], b = X[0];
            for (int k = 0; k <= opt->max_iter; k++) {
                if (X[k] < a) a = X[k];
                if (X[k] > b) b = X[k];
            }

            double c = lb[j], d = ub[j];
            for (int k = 0; k <= opt->max_iter; k++) {
                walks[k * opt->dim + j] = ((X[k] - a) * (d - c)) / (b - a + 1e-10) + c;
            }

            free(X);
        }

        // Store walks for this antlion (simplified for current iteration)
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = walks[t * opt->dim + j];
        }

        free(lb);
        free(ub);
    }

    free(walks);
}

// Update ant positions based on random walks
void update_ant_positions(Optimizer *opt, int t, double *antlion_positions, int antlion_size) {
    double *weights = (double *)malloc(antlion_size * sizeof(double));
    if (!weights) return;

    // Compute roulette wheel weights
    for (int i = 0; i < antlion_size; i++) {
        weights[i] = 1.0 / (opt->population[i].fitness + ROULETTE_EPSILON);
    }

    for (int i = 0; i < opt->population_size; i++) {
        // Select antlion using roulette wheel
        int roulette_idx = roulette_wheel_selection_alo(weights, antlion_size);

        // Random walk around selected antlion
        random_walk_phase(opt, t, &antlion_positions[roulette_idx * opt->dim], 1);
        double *RA = (double *)malloc(opt->dim * sizeof(double));
        if (!RA) {
            free(weights);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            RA[j] = antlion_positions[roulette_idx * opt->dim + j];
        }

        // Random walk around elite antlion
        random_walk_phase(opt, t, opt->best_solution.position, 1);
        double *RE = (double *)malloc(opt->dim * sizeof(double));
        if (!RE) {
            free(RA); free(weights);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            RE[j] = opt->best_solution.position[j];
        }

        // Update ant position (Equation 2.13)
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = (RA[j] + RE[j]) / 2.0;
        }

        free(RA);
        free(RE);
    }

    free(weights);
    enforce_bound_constraints(opt);
}

// Update antlions by combining and sorting populations
void update_antlions_phase(Optimizer *opt, double *antlion_positions, int antlion_size) {
    // Combine antlion and ant populations
    int total_size = antlion_size + opt->population_size;
    double *combined_positions = (double *)malloc(total_size * opt->dim * sizeof(double));
    double *combined_fitness = (double *)malloc(total_size * sizeof(double));
    int *indices = (int *)malloc(total_size * sizeof(int));
    if (!combined_positions || !combined_fitness || !indices) {
        free(combined_positions); free(combined_fitness); free(indices);
        return;
    }

    // Copy antlion positions and fitness
    for (int i = 0; i < antlion_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            combined_positions[i * opt->dim + j] = antlion_positions[i * opt->dim + j];
        }
        combined_fitness[i] = opt->population[i].fitness;
        indices[i] = i;
    }

    // Copy ant positions and compute fitness
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            combined_positions[(antlion_size + i) * opt->dim + j] = opt->population[i].position[j];
        }
        combined_fitness[antlion_size + i] = opt->population[i].fitness;
        indices[antlion_size + i] = antlion_size + i;
    }

    // Sort combined fitness
    for (int i = 0; i < total_size - 1; i++) {
        for (int j = 0; j < total_size - i - 1; j++) {
            if (combined_fitness[j] > combined_fitness[j + 1]) {
                double temp_fitness = combined_fitness[j];
                combined_fitness[j] = combined_fitness[j + 1];
                combined_fitness[j + 1] = temp_fitness;

                int temp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_idx;
            }
        }
    }

    // Update antlion positions
    for (int i = 0; i < antlion_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = combined_positions[indices[i] * opt->dim + j];
        }
        opt->population[i].fitness = combined_fitness[i];
    }

    // Update elite if better solution found
    if (combined_fitness[0] < opt->best_solution.fitness) {
        opt->best_solution.fitness = combined_fitness[0];
        for (int j = 0; j < opt->dim; j++) {
            opt->best_solution.position[j] = combined_positions[indices[0] * opt->dim + j];
        }
    }

    // Ensure elite is in population
    for (int j = 0; j < opt->dim; j++) {
        antlion_positions[j] = opt->best_solution.position[j];
    }
    opt->population[0].fitness = opt->best_solution.fitness;

    free(combined_positions);
    free(combined_fitness);
    free(indices);
}

// Main Optimization Function
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize antlion population
    double *antlion_positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    if (!antlion_positions) return;

    initialize_populations(opt);

    // Copy initial population to antlions
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            antlion_positions[i * opt->dim + j] = opt->population[i].position[j];
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Set initial elite
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Main optimization loop
    for (int t = 0; t < opt->max_iter; t++) {
        update_ant_positions(opt, t, antlion_positions, opt->population_size);
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
        update_antlions_phase(opt, antlion_positions, opt->population_size);

        // Log progress every 50 iterations
        if ((t + 1) % 50 == 0) {
            printf("At iteration %d, the elite fitness is %f\n", t + 1, opt->best_solution.fitness);
        }
    }

    free(antlion_positions);
}
