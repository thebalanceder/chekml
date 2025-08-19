#include "CA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() to seed random generator

// Static alpha variable
static double alpha = 0.5;

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Calculate atmospheric absorption coefficient
double CoefCalculate(double F, double T) {
    double pres = 1.0;  // Atmospheric pressure (atm)
    double relh = 50.0;  // Relative humidity (%)
    double freq_hum = F;
    double temp = T + 273.0;  // Convert to Kelvin

    // Calculate humidity
    double C_humid = 4.6151 - 6.8346 * pow(273.15 / temp, 1.261);
    double hum = relh * pow(10.0, C_humid) * pres;

    // Temperature ratio
    double tempr = temp / 293.15;

    // Oxygen and nitrogen relaxation frequencies
    double frO = pres * (24.0 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum));
    double frN = pres * pow(tempr, -0.5) * (9.0 + 280.0 * hum * exp(-4.17 * (pow(tempr, -1.0/3.0) - 1.0)));

    // Absorption coefficient calculation
    double alpha = 8.686 * freq_hum * freq_hum * (
        1.84e-11 * (1.0 / pres) * sqrt(tempr) +
        pow(tempr, -2.5) * (
            0.01275 * (exp(-2239.1 / temp) * 1.0 / (frO + freq_hum * freq_hum / frO)) +
            0.1068 * (exp(-3352.0 / temp) * 1.0 / (frN + freq_hum * freq_hum / frN))
        )
    );

    // Round to 3 decimal places
    double db_humi = round(alpha * 1000.0) / 1000.0;

    return db_humi;
}

// Cricket Update Phase (combines frequency calculation, movement, and solution update)
void cricket_update_phase(Optimizer *opt) {
    double *Q = (double *)calloc(opt->population_size, sizeof(double));
    double **v = (double **)calloc(opt->population_size, sizeof(double *));
    for (int i = 0; i < opt->population_size; i++) {
        v[i] = (double *)calloc(opt->dim, sizeof(double));
    }
    double *scale = (double *)calloc(opt->dim, sizeof(double));
    for (int j = 0; j < opt->dim; j++) {
        scale[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
    }

    for (int i = 0; i < opt->population_size; i++) {
        // Simulate cricket parameters
        double *N = (double *)calloc(opt->dim, sizeof(double));
        double *T = (double *)calloc(opt->dim, sizeof(double));
        double *C = (double *)calloc(opt->dim, sizeof(double));
        double *V = (double *)calloc(opt->dim, sizeof(double));
        double *Z = (double *)calloc(opt->dim, sizeof(double));
        double *F = (double *)calloc(opt->dim, sizeof(double));

        for (int j = 0; j < opt->dim; j++) {
            N[j] = (double)(rand() % 121);  // Random integer [0, 120]
            T[j] = TEMP_COEFF * N[j] + TEMP_OFFSET;
            T[j] = (T[j] < TEMP_MIN) ? TEMP_MIN : (T[j] > TEMP_MAX) ? TEMP_MAX : T[j];
            C[j] = (5.0 / 9.0) * (T[j] - 32.0);
            V[j] = 20.1 * sqrt(273.0 + C[j]);
            V[j] = sqrt(V[j]) / 1000.0;
            Z[j] = opt->population[i].position[j] - opt->best_solution.position[j];
            F[j] = (Z[j] != 0.0) ? V[j] / Z[j] : 0.0;
        }

        // Compute Q[i] as scalar (mean of F)
        double F_mean = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            F_mean += F[j];
        }
        F_mean /= opt->dim;
        Q[i] = Q_MIN + (F_mean - Q_MIN) * rand_double(0.0, 1.0);

        // Update velocity and position (S)
        double *S = (double *)calloc(opt->dim, sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            v[i][j] += (opt->population[i].position[j] - opt->best_solution.position[j]) * Q[i] + V[j];
            S[j] = opt->population[i].position[j] + v[i][j];
        }

        // Calculate gamma
        double SumF = F_mean + FREQ_OFFSET;
        double SumT = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            SumT += C[j];
        }
        SumT /= opt->dim;
        double gamma = CoefCalculate(SumF, SumT);

        // Update solution based on fitness comparison
        double *M = (double *)calloc(opt->dim, sizeof(double));
        for (int k = 0; k < opt->population_size; k++) {
            if (opt->population[i].fitness < opt->population[k].fitness) {
                double distance = 0.0;
                for (int j = 0; j < opt->dim; j++) {
                    distance += pow(opt->population[i].position[j] - opt->population[k].position[j], 2);
                }
                distance = sqrt(distance);
                double PS = opt->population[i].fitness * (4.0 * PI * (distance * distance));
                double Lp = PS + 10.0 * log10(1.0 / (4.0 * PI * (distance * distance)));
                double Aatm = (7.4 * (F_mean * F_mean * distance) / (50.0 * pow(10.0, -8.0)));
                double RLP = Lp - Aatm;
                double K = RLP * exp(-gamma * distance * distance);
                double beta = K + BETA_MIN;

                for (int j = 0; j < opt->dim; j++) {
                    double tmpf = alpha * (rand_double(0.0, 1.0) - 0.5) * scale[j];
                    M[j] = opt->population[i].position[j] * (1.0 - beta) + opt->population[k].position[j] * beta + tmpf;
                }
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    M[j] = opt->best_solution.position[j] + 0.01 * (rand_double(0.0, 1.0) - 0.5);
                }
            }
        }

        // Select new solution
        double *new_position = (rand_double(0.0, 1.0) > gamma) ? S : M;

        // Update population
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = new_position[j];
        }

        // Free temporary arrays
        free(N);
        free(T);
        free(C);
        free(V);
        free(Z);
        free(F);
        free(S);
        free(M);
    }

    // Update alpha
    double delta = 0.97;
    alpha *= delta;

    // Free allocated memory
    for (int i = 0; i < opt->population_size; i++) {
        free(v[i]);
    }
    free(v);
    free(Q);
    free(scale);

    enforce_bound_constraints(opt);
}

// Main Optimization Function
void CA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));  // Seed random number generator
    alpha = 0.5;  // Initialize alpha
    int iter = 0;
    while (opt->best_solution.fitness > TOL && iter < MAX_ITER_CA) {
        cricket_update_phase(opt);

        // Evaluate fitness and update best solution
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
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