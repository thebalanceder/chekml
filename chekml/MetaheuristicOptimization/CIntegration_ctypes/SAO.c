/* SAO.c - Implementation file for Snow Ablation Optimization */
#include "SAO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void initialize_population_sao(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = LOW + rand_double(0.0, 1.0) * (UP - LOW);
        }
        opt->population[i].fitness = INFINITY; // Will be updated by objective function
    }
    enforce_sao_bound_constraints(opt);
}

// Simulate Brownian Motion
void brownian_motion(Optimizer *opt, double **motion, int num_pop) {
    for (int i = 0; i < num_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            // Approximate normal distribution using Box-Muller transform
            double u1 = rand_double(0.0, 1.0);
            double u2 = rand_double(0.0, 1.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            motion[i][j] = z * BROWNIAN_VARIANCE;
        }
    }
}

// Calculate Centroid of Population
void calculate_centroid(Optimizer *opt, double *centroid) {
    for (int j = 0; j < opt->dim; j++) {
        centroid[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            centroid[j] += opt->population[i].position[j];
        }
        centroid[j] /= opt->population_size;
    }
}

// Select Elite Individual
void select_elite(Optimizer *opt, double *elite, int *best_idx) {
    int indices[NUM_ELITES];
    double fitness[NUM_ELITES];
    for (int i = 0; i < NUM_ELITES; i++) {
        indices[i] = i;
        fitness[i] = opt->population[i].fitness;
    }

    // Simple sort to find top NUM_ELITES
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < NUM_ELITES; j++) {
            if (opt->population[i].fitness < fitness[j]) {
                for (int k = NUM_ELITES - 1; k > j; k--) {
                    fitness[k] = fitness[k - 1];
                    indices[k] = indices[k - 1];
                }
                fitness[j] = opt->population[i].fitness;
                indices[j] = i;
                break;
            }
        }
    }

    // Rank-based probabilities
    double ranks[NUM_ELITES] = {4.0, 3.0, 2.0, 1.0};
    double sum_ranks = 10.0; // Sum of ranks
    double probs[NUM_ELITES];
    for (int i = 0; i < NUM_ELITES; i++) {
        probs[i] = ranks[i] / sum_ranks;
    }

    // Select one elite randomly based on probabilities
    double r = rand_double(0.0, 1.0);
    double cum_prob = 0.0;
    int selected_idx = indices[0];
    for (int i = 0; i < NUM_ELITES; i++) {
        cum_prob += probs[i];
        if (r <= cum_prob) {
            selected_idx = indices[i];
            break;
        }
    }

    for (int j = 0; j < opt->dim; j++) {
        elite[j] = opt->population[selected_idx].position[j];
    }
    *best_idx = indices[0]; // Best individual index
}

// Covariance Matrix Learning (Simplified)
void covariance_matrix_learning(Optimizer *opt, double **QQ) {
    // Compute mean
    double *mean = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            mean[j] += opt->population[i].position[j];
        }
        mean[j] /= opt->population_size;
    }

    // Compute covariance matrix (simplified to diagonal for efficiency)
    double *cov_diag = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        cov_diag[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            double diff = opt->population[i].position[j] - mean[j];
            cov_diag[j] += diff * diff;
        }
        cov_diag[j] = (cov_diag[j] / (opt->population_size - 1)) + 1e-6;
    }

    // Apply diagonal scaling (approximating eigenvectors)
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            QQ[i][j] = opt->population[i].position[j] * sqrt(cov_diag[j]);
        }
    }

    free(mean);
    free(cov_diag);
}

// Historical Boundary Adjustment
void historical_boundary_adjustment(double *position, int dim) {
    for (int j = 0; j < dim; j++) {
        if (position[j] < LOW) {
            position[j] = LOW;
        } else if (position[j] > UP) {
            position[j] = UP;
        }
    }
}

// Random Centroid Reverse Learning
void random_centroid_reverse_learning(Optimizer *opt, double **reverse_pop) {
    int B = 2 + (rand() % ((opt->population_size / 2) - 1));
    int *indices = (int *)malloc(B * sizeof(int));
    for (int i = 0; i < B; i++) {
        indices[i] = rand() % opt->population_size;
    }

    // Compute centroid of selected individuals
    double *centroid = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        centroid[j] = 0.0;
        for (int i = 0; i < B; i++) {
            centroid[j] += opt->population[indices[i]].position[j];
        }
        centroid[j] /= B;
    }

    // Generate reverse population
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            reverse_pop[i][j] = 2.0 * centroid[j] - opt->population[i].position[j];
            if (reverse_pop[i][j] < LOW) reverse_pop[i][j] = LOW;
            if (reverse_pop[i][j] > UP) reverse_pop[i][j] = UP;
        }
    }

    free(indices);
    free(centroid);
}

// Calculate Snow Ablation Rate
double calculate_snow_ablation_rate(int iter, int max_iter) {
    double T = exp(-((double)iter / max_iter));
    double Df = DF_MIN + (DF_MAX - DF_MIN) * (exp((double)iter / max_iter) - 1.0) / (EULER - 1.0);
    return Df * T;
}

// Enforce Boundary Constraints
void enforce_sao_bound_constraints(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        historical_boundary_adjustment(opt->population[i].position, opt->dim);
    }
}

// Main Optimization Function
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(42); // Set random seed for reproducibility

    // Initialize population
    initialize_population_sao(opt);
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    // Allocate temporary arrays
    double *centroid = (double *)malloc(opt->dim * sizeof(double));
    double *elite = (double *)malloc(opt->dim * sizeof(double));
    double **brownian = (double **)malloc(opt->population_size * sizeof(double *));
    double **QQ = (double **)malloc(opt->population_size * sizeof(double *));
    double **reverse_pop = (double **)malloc(opt->population_size * sizeof(double *));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *reverse_fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *combined_fitness = (double *)malloc(2 * opt->population_size * sizeof(double));
    int *indices = (int *)malloc(2 * opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        brownian[i] = (double *)malloc(opt->dim * sizeof(double));
        QQ[i] = (double *)malloc(opt->dim * sizeof(double));
        reverse_pop[i] = (double *)malloc(opt->dim * sizeof(double));
    }

    int num_a = opt->population_size / 2;
    int num_b = opt->population_size - num_a;
    int best_idx;

    for (int iter = 0; iter < SAO_MAX_ITER; iter++) {
        double R = calculate_snow_ablation_rate(iter, SAO_MAX_ITER);

        // Exploration phase
        brownian_motion(opt, brownian, num_a);
        for (int i = 0; i < num_a; i++) {
            select_elite(opt, elite, &best_idx);
            calculate_centroid(opt, centroid);
            double alpha1 = rand_double(0.0, 1.0);
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] = elite[j] + R * brownian[i][j] * (
                    alpha1 * (opt->best_solution.position[j] - opt->population[i].position[j]) +
                    (1.0 - alpha1) * (centroid[j] - opt->population[i].position[j])
                );
            }
            historical_boundary_adjustment(opt->population[i].position, opt->dim);
        }

        // Development phase
        covariance_matrix_learning(opt, QQ);
        for (int i = num_a; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                opt->population[i].position[j] += R * QQ[i][j];
            }
            historical_boundary_adjustment(opt->population[i].position, opt->dim);
        }

        // Update fitness
        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }

        // Random centroid reverse learning
        random_centroid_reverse_learning(opt, reverse_pop);
        for (int i = 0; i < opt->population_size; i++) {
            reverse_fitness[i] = objective_function(reverse_pop[i]);
        }

        // Greedy selection
        for (int i = 0; i < opt->population_size; i++) {
            combined_fitness[i] = fitness[i];
            combined_fitness[i + opt->population_size] = reverse_fitness[i];
            indices[i] = i;
            indices[i + opt->population_size] = i + opt->population_size;
        }

        // Sort indices by fitness
        for (int i = 0; i < 2 * opt->population_size - 1; i++) {
            for (int j = i + 1; j < 2 * opt->population_size; j++) {
                if (combined_fitness[indices[i]] > combined_fitness[indices[j]]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }

        // Update population
        for (int i = 0; i < opt->population_size; i++) {
            int idx = indices[i];
            if (idx < opt->population_size) {
                // Keep original
                fitness[i] = combined_fitness[idx];
            } else {
                // Take from reverse_pop
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = reverse_pop[idx - opt->population_size][j];
                }
                fitness[i] = combined_fitness[idx];
            }
        }

        enforce_sao_bound_constraints(opt);

        // Adjust subpopulation sizes
        if (num_a < opt->population_size) {
            num_a++;
            num_b--;
        }
    }

    // Free allocated memory
    free(centroid);
    free(elite);
    free(fitness);
    free(reverse_fitness);
    free(combined_fitness);
    free(indices);
    for (int i = 0; i < opt->population_size; i++) {
        free(brownian[i]);
        free(QQ[i]);
        free(reverse_pop[i]);
    }
    free(brownian);
    free(QQ);
    free(reverse_pop);
}
