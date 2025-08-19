/* SAO.c - Optimized Implementation file for Snow Ablation Optimization */
#include "SAO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Function to generate a random double between min and max
inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Initialize Population
inline void initialize_population(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = LOW + rand_double(0.0, 1.0) * (UP - LOW);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_sao_bound_constraints(opt);
}

// Simulate Brownian Motion
inline void brownian_motion(Optimizer *opt, double **motion, int num_pop) {
    for (int i = 0; i < num_pop; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double u1 = rand_double(0.0, 1.0);
            double u2 = rand_double(0.0, 1.0);
            motion[i][j] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * BROWNIAN_VARIANCE;
        }
    }
}

// Calculate Centroid of Population
inline void calculate_centroid(Optimizer *opt, double *centroid) {
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            sum += opt->population[i].position[j];
        }
        centroid[j] = sum / opt->population_size;
    }
}

// Select Elite Individual
inline void select_elite(Optimizer *opt, double *elite, int *best_idx) {
    int indices[NUM_ELITES];
    double fitness[NUM_ELITES];
    for (int i = 0; i < NUM_ELITES; i++) {
        indices[i] = i;
        fitness[i] = opt->population[i].fitness;
    }

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

    static const double ranks[NUM_ELITES] = {4.0, 3.0, 2.0, 1.0};
    static const double sum_ranks = 10.0;
    double r = rand_double(0.0, 1.0);
    double cum_prob = 0.0;
    int selected_idx = indices[0];
    for (int i = 0; i < NUM_ELITES; i++) {
        cum_prob += ranks[i] / sum_ranks;
        if (r <= cum_prob) {
            selected_idx = indices[i];
            break;
        }
    }

    memcpy(elite, opt->population[selected_idx].position, opt->dim * sizeof(double));
    *best_idx = indices[0];
}

// Covariance Matrix Learning (Simplified)
inline void covariance_matrix_learning(Optimizer *opt, double **QQ) {
    double *mean = (double *)malloc(opt->dim * sizeof(double));
    double *cov_diag = (double *)malloc(opt->dim * sizeof(double));

    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            sum += opt->population[i].position[j];
        }
        mean[j] = sum / opt->population_size;
        cov_diag[j] = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            double diff = opt->population[i].position[j] - mean[j];
            cov_diag[j] += diff * diff;
        }
        cov_diag[j] = sqrt(cov_diag[j] / (opt->population_size - 1) + 1e-6);
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            QQ[i][j] = opt->population[i].position[j] * cov_diag[j];
        }
    }

    free(mean);
    free(cov_diag);
}

// Historical Boundary Adjustment
inline void historical_boundary_adjustment(double *position, int dim) {
    for (int j = 0; j < dim; j++) {
        position[j] = fmax(LOW, fmin(UP, position[j]));
    }
}

// Random Centroid Reverse Learning
inline void random_centroid_reverse_learning(Optimizer *opt, double **reverse_pop) {
    int B = 2 + (rand() % ((opt->population_size / 2) - 1));
    int *indices = (int *)malloc(B * sizeof(int));
    for (int i = 0; i < B; i++) {
        indices[i] = rand() % opt->population_size;
    }

    double *centroid = (double *)malloc(opt->dim * sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < B; i++) {
            sum += opt->population[indices[i]].position[j];
        }
        centroid[j] = sum / B;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            reverse_pop[i][j] = fmax(LOW, fmin(UP, 2.0 * centroid[j] - opt->population[i].position[j]));
        }
    }

    free(indices);
    free(centroid);
}

// Calculate Snow Ablation Rate
inline double calculate_snow_ablation_rate(int iter, int max_iter) {
    double t = (double)iter / max_iter;
    double T = exp(-t);
    double Df = DF_MIN + (DF_MAX - DF_MIN) * (exp(t) - 1.0) / (EULER - 1.0);
    return Df * T;
}

// Enforce Boundary Constraints
inline void enforce_sao_bound_constraints(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        historical_boundary_adjustment(opt->population[i].position, opt->dim);
    }
}

// Comparison function for qsort
static int compare_fitness(const void *a, const void *b) {
    double fa = ((const double *)a)[0];
    double fb = ((const double *)b)[0];
    return (fa > fb) - (fa < fb);
}

// Main Optimization Function
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(42); // Set random seed for reproducibility

    // Initialize population
    initialize_population(opt);
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    // Allocate temporary arrays
    double centroid[opt->dim]; // Stack-allocated for small dim
    double elite[opt->dim];   // Stack-allocated for small dim
    double **brownian = (double **)malloc(opt->population_size * sizeof(double *));
    double **QQ = (double **)malloc(opt->population_size * sizeof(double *));
    double **reverse_pop = (double **)malloc(opt->population_size * sizeof(double *));
    double *fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *reverse_fitness = (double *)malloc(opt->population_size * sizeof(double));
    double *combined_fitness = (double *)malloc(2 * opt->population_size * sizeof(double));
    int *indices = (int *)malloc(2 * opt->population_size * sizeof(int));
    double *sort_fitness = (double *)malloc(2 * opt->population_size * sizeof(double));
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
                double delta_best = opt->best_solution.position[j] - opt->population[i].position[j];
                double delta_centroid = centroid[j] - opt->population[i].position[j];
                opt->population[i].position[j] = elite[j] + R * brownian[i][j] * (
                    alpha1 * delta_best + (1.0 - alpha1) * delta_centroid
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
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        // Random centroid reverse learning
        random_centroid_reverse_learning(opt, reverse_pop);
        for (int i = 0; i < opt->population_size; i++) {
            reverse_fitness[i] = objective_function(reverse_pop[i]);
        }

        // Greedy selection using qsort
        for (int i = 0; i < opt->population_size; i++) {
            combined_fitness[i] = fitness[i];
            combined_fitness[i + opt->population_size] = reverse_fitness[i];
            indices[i] = i;
            indices[i + opt->population_size] = i + opt->population_size;
        }

        for (int i = 0; i < 2 * opt->population_size; i++) {
            sort_fitness[i] = combined_fitness[i];
        }
        qsort(sort_fitness, 2 * opt->population_size, sizeof(double), compare_fitness);

        // Update population
        for (int i = 0; i < opt->population_size; i++) {
            int idx = indices[i];
            if (combined_fitness[idx] <= sort_fitness[i]) {
                fitness[i] = combined_fitness[idx];
            } else {
                for (int j = 0; j < opt->population_size; j++) {
                    if (combined_fitness[j + opt->population_size] <= sort_fitness[i]) {
                        memcpy(opt->population[i].position, reverse_pop[j], opt->dim * sizeof(double));
                        fitness[i] = reverse_fitness[j];
                        break;
                    }
                }
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
    free(fitness);
    free(reverse_fitness);
    free(combined_fitness);
    free(indices);
    free(sort_fitness);
    for (int i = 0; i < opt->population_size; i++) {
        free(brownian[i]);
        free(QQ[i]);
        free(reverse_pop[i]);
    }
    free(brownian);
    free(QQ);
    free(reverse_pop);
}
