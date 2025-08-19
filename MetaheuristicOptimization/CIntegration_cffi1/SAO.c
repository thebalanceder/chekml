/* SAO.c - Ultra-Optimized Implementation for Snow Ablation Optimization */
#include "SAO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Inline random double in [min, max] */
double rand_double(double min, double max);

/* Initialize population with vectorizable loop */
inline void initialize_population_sao(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = -5.0 + 10.0 * ((double)rand() / RAND_MAX); // [LOW, UP]
        }
        opt->population[i].fitness = INFINITY;
    }
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = fmax(-5.0, fmin(5.0, pos[j]));
        }
    }
}

/* Brownian motion with precomputed constants */
inline void brownian_motion(double **motion, int num_pop, int dim) {
    const double c = 0.5 * 0.7071067811865475; // BROWNIAN_VARIANCE * sqrt(0.5)
    for (int i = 0; i < num_pop; i++) {
        double *m = motion[i];
        for (int j = 0; j < dim; j++) {
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            m[j] = c * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
}

/* Centroid calculation with single-pass accumulation */
inline void calculate_centroid(Optimizer *opt, double *centroid) {
    double scale = 1.0 / opt->population_size;
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            sum += opt->population[i].position[j];
        }
        centroid[j] = sum * scale;
    }
}

/* Elite selection with fixed-size arrays */
inline void select_elite(Optimizer *opt, double *elite, int *best_idx) {
    double fitness[4] = {opt->population[0].fitness, opt->population[1].fitness,
                         opt->population[2].fitness, opt->population[3].fitness};
    int indices[4] = {0, 1, 2, 3};

    for (int i = 0; i < opt->population_size; i++) {
        double f = opt->population[i].fitness;
        for (int j = 0; j < 4; j++) {
            if (f < fitness[j]) {
                for (int k = 3; k > j; k--) {
                    fitness[k] = fitness[k - 1];
                    indices[k] = indices[k - 1];
                }
                fitness[j] = f;
                indices[j] = i;
                break;
            }
        }
    }

    static const double probs[4] = {0.4, 0.3, 0.2, 0.1}; // Precomputed 4/10, 3/10, 2/10, 1/10
    double r = (double)rand() / RAND_MAX;
    double cum = 0.0;
    int sel = indices[0];
    for (int i = 0; i < 4; i++) {
        cum += probs[i];
        if (r <= cum) {
            sel = indices[i];
            break;
        }
    }

    memcpy(elite, opt->population[sel].position, opt->dim * sizeof(double));
    *best_idx = indices[0];
}

/* Simplified covariance matrix learning */
inline void covariance_matrix_learning(Optimizer *opt, double **QQ) {
    double mean[opt->dim];
    double scale = 1.0 / opt->population_size;
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            sum += opt->population[i].position[j];
        }
        mean[j] = sum * scale;
    }

    for (int j = 0; j < opt->dim; j++) {
        double var = 0.0;
        for (int i = 0; i < opt->population_size; i++) {
            double d = opt->population[i].position[j] - mean[j];
            var += d * d;
        }
        var = sqrt(var * scale + 1e-6);
        for (int i = 0; i < opt->population_size; i++) {
            QQ[i][j] = opt->population[i].position[j] * var;
        }
    }
}

/* Boundary adjustment with arithmetic operations */
inline void historical_boundary_adjustment(double *pos, int dim) {
    for (int j = 0; j < dim; j++) {
        pos[j] = fmax(-5.0, fmin(5.0, pos[j]));
    }
}

/* Reverse learning with minimal allocations */
inline void random_centroid_reverse_learning(Optimizer *opt, double **reverse_pop) {
    int B = 2 + (rand() % (opt->population_size / 2 - 1));
    int idx[B];
    for (int i = 0; i < B; i++) {
        idx[i] = rand() % opt->population_size;
    }

    double centroid[opt->dim];
    double scale = 1.0 / B;
    for (int j = 0; j < opt->dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < B; i++) {
            sum += opt->population[idx[i]].position[j];
        }
        centroid[j] = sum * scale;
    }

    for (int i = 0; i < opt->population_size; i++) {
        double *rp = reverse_pop[i];
        double *p = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            rp[j] = fmax(-5.0, fmin(5.0, 2.0 * centroid[j] - p[j]));
        }
    }
}

/* Precomputed snow ablation rate */
inline double calculate_snow_ablation_rate(int iter) {
    double t = (double)iter * 0.001; // 1/SAO_MAX_ITER
    double T = exp(-t);
    double Df = 0.35 + 0.25 * (exp(t) - 1.0) * 0.582240524; // (DF_MAX - DF_MIN)/(e-1)
    return Df * T;
}

/* Enforce constraints in bulk */
inline void enforce_sao_bound_constraints(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        historical_boundary_adjustment(opt->population[i].position, opt->dim);
    }
}

/* qsort comparison for fitness */
static int compare_fitness(const void *a, const void *b) {
    double fa = *(const double *)a;
    double fb = *(const double *)b;
    return (fa > fb) - (fa < fb);
}

/* Main optimization loop */
void SAO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(42);

    double fitness[50]; // NUM_POP
    double reverse_fitness[50];
    double combined_fitness[100]; // 2 * NUM_POP
    int indices[100];
    double *brownian[50]; // NUM_POP
    double *QQ[50];
    double *reverse_pop[50];
    double centroid[opt->dim];
    double elite[opt->dim];

    // Allocate arrays
    for (int i = 0; i < 50; i++) {
        brownian[i] = (double *)malloc(opt->dim * sizeof(double));
        QQ[i] = (double *)malloc(opt->dim * sizeof(double));
        reverse_pop[i] = (double *)malloc(opt->dim * sizeof(double));
    }

    initialize_population_sao(opt);
    for (int i = 0; i < opt->population_size; i++) {
        fitness[i] = objective_function(opt->population[i].position);
        if (fitness[i] < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness[i];
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    int num_a = opt->population_size >> 1;
    int num_b = opt->population_size - num_a;
    int best_idx;

    for (int iter = 0; iter < 1000; iter++) { // SAO_MAX_ITER
        double R = calculate_snow_ablation_rate(iter);

        brownian_motion(brownian, num_a, opt->dim);
        for (int i = 0; i < num_a; i++) {
            select_elite(opt, elite, &best_idx);
            calculate_centroid(opt, centroid);
            double a = (double)rand() / RAND_MAX;
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] = elite[j] + R * brownian[i][j] * (
                    a * (opt->best_solution.position[j] - pos[j]) +
                    (1.0 - a) * (centroid[j] - pos[j])
                );
            }
            historical_boundary_adjustment(pos, opt->dim);
        }

        covariance_matrix_learning(opt, QQ);
        for (int i = num_a; i < opt->population_size; i++) {
            double *pos = opt->population[i].position;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += R * QQ[i][j];
            }
            historical_boundary_adjustment(pos, opt->dim);
        }

        for (int i = 0; i < opt->population_size; i++) {
            fitness[i] = objective_function(opt->population[i].position);
            if (fitness[i] < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness[i];
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        random_centroid_reverse_learning(opt, reverse_pop);
        for (int i = 0; i < opt->population_size; i++) {
            reverse_fitness[i] = objective_function(reverse_pop[i]);
        }

        for (int i = 0; i < opt->population_size; i++) {
            combined_fitness[i] = fitness[i];
            combined_fitness[i + opt->population_size] = reverse_fitness[i];
            indices[i] = i;
            indices[i + opt->population_size] = i + opt->population_size;
        }

        qsort(combined_fitness, 2 * opt->population_size, sizeof(double), compare_fitness);
        for (int i = 0; i < opt->population_size; i++) {
            int idx = indices[i];
            if (idx < opt->population_size) {
                fitness[i] = combined_fitness[i];
            } else {
                memcpy(opt->population[i].position, reverse_pop[idx - opt->population_size], opt->dim * sizeof(double));
                fitness[i] = combined_fitness[i];
            }
        }

        enforce_sao_bound_constraints(opt);

        if (num_a < opt->population_size) {
            num_a++;
            num_b--;
        }
    }

    // Free allocated memory
    for (int i = 0; i < 50; i++) {
        free(brownian[i]);
        free(QQ[i]);
        free(reverse_pop[i]);
    }
}
