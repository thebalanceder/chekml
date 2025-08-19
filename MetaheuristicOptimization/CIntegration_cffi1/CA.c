#include "CA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator
#include <emmintrin.h>  // For SSE intrinsics
#include <omp.h>       // For OpenMP parallelization
#include <string.h>    // For memcpy
#include <errno.h>     // For errno and ENOMEM

// Static alpha variable
static double alpha = 0.5;

// Thread-local random number generator state
static __thread unsigned int rand_seed;

// Initialize thread-local seed
void init_rand_seed() {
    rand_seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
}

// Fast random double between min and max (inline for performance)
inline double CA_rand_double(double min, double max) {
    rand_seed = rand_seed * 1103515245 + 12345;  // Linear Congruential Generator
    return min + (max - min) * ((double)(rand_seed >> 16) / 0xFFFF);
}

// Fast square root approximation (using SSE if available)
inline double fast_sqrt(double x) {
    if (x < 1e-10) return 0.0;
    __m128d vec = _mm_set1_pd(x);
    vec = _mm_sqrt_pd(vec);
    return _mm_cvtsd_f64(vec);
}

// Calculate atmospheric absorption coefficient (optimized for fewer math calls)
double CA_CoefCalculate(double F, double T) {
    const double pres = 1.0;  // Atmospheric pressure (atm)
    const double relh = 50.0;  // Relative humidity (%)
    double freq_hum = F;
    double temp = T + 273.0;  // Convert to Kelvin

    // Avoid division by zero
    if (temp < 1e-10) temp = 1e-10;

    // Calculate humidity
    double temp_ratio = 273.15 / temp;
    double C_humid = 4.6151 - 6.8346 * pow(temp_ratio, 1.261);
    double hum = relh * pow(10.0, C_humid) * pres;

    // Temperature ratio
    double tempr = temp / 293.15;
    double sqrt_tempr = fast_sqrt(tempr);

    // Oxygen and nitrogen relaxation frequencies
    double frO = pres * (24.0 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum));
    double frN = pres * sqrt_tempr * (9.0 + 280.0 * hum * exp(-4.17 * (pow(tempr, -1.0/3.0) - 1.0)));

    // Absorption coefficient calculation
    double freq_hum_sq = freq_hum * freq_hum;
    double inv_temp = 1.0 / temp;
    double tempr_inv_2_5 = tempr * tempr * sqrt_tempr;  // 1/tempr^2.5
    double alpha = 8.686 * freq_hum_sq * (
        1.84e-11 * (1.0 / pres) * sqrt_tempr +
        tempr_inv_2_5 * (
            0.01275 * (exp(-2239.1 * inv_temp) / (frO + freq_hum_sq / frO)) +
            0.1068 * (exp(-3352.0 * inv_temp) / (frN + freq_hum_sq / frN))
        )
    );

    // Round to 3 decimal places
    return round(alpha * 1000.0) / 1000.0;
}

// Cricket Update Phase (optimized for CPU with OpenMP and cache efficiency)
void CA_cricket_update_phase(Optimizer *opt) {
    // Pre-allocate reusable buffers (aligned for cache)
    double *Q = NULL, **v = NULL, *scale = NULL;
    double *N = NULL, *T = NULL, *C = NULL, *V = NULL, *Z = NULL, *F = NULL, *S = NULL, *M = NULL;
    int alloc_failed = 0;

    // Allocate memory with error checking
    if (posix_memalign((void **)&Q, 64, opt->population_size * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&v, 64, opt->population_size * sizeof(double *)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&scale, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&N, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&T, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&C, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&V, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&Z, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&F, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&S, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }
    if (posix_memalign((void **)&M, 64, opt->dim * sizeof(double)) != 0) {
        alloc_failed = 1; goto cleanup;
    }

    memset(Q, 0, opt->population_size * sizeof(double));
    for (int i = 0; i < opt->population_size; i++) {
        v[i] = NULL;
        if (posix_memalign((void **)&v[i], 64, opt->dim * sizeof(double)) != 0) {
            alloc_failed = 1; goto cleanup;
        }
        memset(v[i], 0, opt->dim * sizeof(double));
    }
    for (int j = 0; j < opt->dim; j++) {
        scale[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
    }

    #pragma omp parallel
    {
        init_rand_seed();  // Initialize thread-local random seed
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < opt->population_size; i++) {
            // Simulate cricket parameters
            double F_mean = 0.0, SumT = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                N[j] = (double)(rand_seed % 121);  // Random integer [0, 120]
                rand_seed = rand_seed * 1103515245 + 12345;
                T[j] = CA_TEMP_COEFF * N[j] + CA_TEMP_OFFSET;
                T[j] = (T[j] < CA_TEMP_MIN) ? CA_TEMP_MIN : (T[j] > CA_TEMP_MAX) ? CA_TEMP_MAX : T[j];
                C[j] = (5.0 / 9.0) * (T[j] - 32.0);
                SumT += C[j];
                V[j] = 20.1 * fast_sqrt(273.0 + C[j]);
                V[j] = fast_sqrt(V[j]) / 1000.0;
                Z[j] = opt->population[i].position[j] - opt->best_solution.position[j];
                F[j] = (fabs(Z[j]) > 1e-10) ? V[j] / Z[j] : 0.0;
                F_mean += F[j];
            }
            F_mean /= opt->dim;
            SumT /= opt->dim;

            // Compute Q[i]
            Q[i] = CA_Q_MIN + (F_mean - CA_Q_MIN) * CA_rand_double(0.0, 1.0);

            // Update velocity and position (S)
            for (int j = 0; j < opt->dim; j++) {
                v[i][j] += (opt->population[i].position[j] - opt->best_solution.position[j]) * Q[i] + V[j];
                S[j] = opt->population[i].position[j] + v[i][j];
            }

            // Calculate gamma
            double SumF = F_mean + CA_FREQ_OFFSET;
            double gamma = CA_CoefCalculate(SumF, SumT);

            // Update solution based on fitness comparison
            for (int k = 0; k < opt->population_size; k++) {
                if (opt->population[i].fitness < opt->population[k].fitness) {
                    double distance = 0.0;
                    for (int j = 0; j < opt->dim; j++) {
                        double diff = opt->population[i].position[j] - opt->population[k].position[j];
                        distance += diff * diff;
                    }
                    distance = fast_sqrt(distance);
                    double distance_sq = distance * distance;
                    double PS = opt->population[i].fitness * (4.0 * CA_PI * distance_sq);
                    double Lp = PS + 10.0 * log10(1.0 / (4.0 * CA_PI * distance_sq));
                    double Aatm = (7.4 * (F_mean * F_mean * distance) / (50.0 * 1e-8));
                    double RLP = Lp - Aatm;
                    double K = RLP * exp(-gamma * distance_sq);
                    double beta = K + CA_BETA_MIN;

                    for (int j = 0; j < opt->dim; j++) {
                        double tmpf = alpha * (CA_rand_double(0.0, 1.0) - 0.5) * scale[j];
                        M[j] = opt->population[i].position[j] * (1.0 - beta) + opt->population[k].position[j] * beta + tmpf;
                    }
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        M[j] = opt->best_solution.position[j] + 0.01 * (CA_rand_double(0.0, 1.0) - 0.5);
                    }
                }
            }

            // Select new solution
            double *new_position = (CA_rand_double(0.0, 1.0) > gamma) ? S : M;

            // Update population
            memcpy(opt->population[i].position, new_position, opt->dim * sizeof(double));
        }
    }

    // Update alpha
    alpha *= 0.97;

cleanup:
    // Free allocated memory
    for (int i = 0; i < opt->population_size && v != NULL; i++) {
        if (v[i] != NULL) free(v[i]);
    }
    if (v != NULL) free(v);
    if (Q != NULL) free(Q);
    if (scale != NULL) free(scale);
    if (N != NULL) free(N);
    if (T != NULL) free(T);
    if (C != NULL) free(C);
    if (V != NULL) free(V);
    if (Z != NULL) free(Z);
    if (F != NULL) free(F);
    if (S != NULL) free(S);
    if (M != NULL) free(M);

    if (alloc_failed) {
        fprintf(stderr, "Memory allocation failed in CA_cricket_update_phase: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    enforce_bound_constraints(opt);
}

// Main Optimization Function
void CA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));  // Seed random number generator
    alpha = 0.5;  // Initialize alpha
    int iter = 0;
    while (opt->best_solution.fitness > CA_TOL && iter < CA_MAX_ITER) {
        CA_cricket_update_phase(opt);

        // Evaluate fitness and update best solution
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                }
            }
        }

        enforce_bound_constraints(opt);
        iter++;
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    printf("Number of iterations: %d\n", iter);
    printf("Best solution: [");
    for (int j = 0; j < opt->dim - 1; j++) {
        printf("%f, ", opt->best_solution.position[j]);
    }
    printf("%f]\n", opt->best_solution.position[opt->dim - 1]);
    printf("Best fitness: %f\n", opt->best_solution.fitness);
}

// Optimization with History for Benchmarking
void CA_optimize_with_history(Optimizer *opt, double (*objective_function)(double *), double **history, int *history_size, int max_history) {
    srand(time(NULL));  // Seed random number generator
    alpha = 0.5;  // Initialize alpha
    int iter = 0;
    *history_size = 0;

    while (opt->best_solution.fitness > CA_TOL && iter < CA_MAX_ITER) {
        CA_cricket_update_phase(opt);

        // Evaluate fitness and update best solution
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            #pragma omp critical
            {
                if (new_fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = new_fitness;
                    memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                    if (*history_size < max_history) {
                        history[*history_size] = (double *)malloc((opt->dim + 1) * sizeof(double));
                        if (history[*history_size] == NULL) {
                            fprintf(stderr, "Memory allocation failed for history: %s\n", strerror(errno));
                            exit(EXIT_FAILURE);
                        }
                        memcpy(history[*history_size], opt->best_solution.position, opt->dim * sizeof(double));
                        history[*history_size][opt->dim] = opt->best_solution.fitness;
                        (*history_size)++;
                    }
                }
            }
        }

        enforce_bound_constraints(opt);
        iter++;
        printf("Iteration %d: Best Fitness = %f\n", iter, opt->best_solution.fitness);
    }

    printf("Number of iterations: %d\n", iter);
    printf("Best solution: [");
    for (int j = 0; j < opt->dim - 1; j++) {
        printf("%f, ", opt->best_solution.position[j]);
    }
    printf("%f]\n", opt->best_solution.position[opt->dim - 1]);
    printf("Best fitness: %f\n", opt->best_solution.fitness);
}
