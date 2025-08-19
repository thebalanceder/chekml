#include "WO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random double between min and max
double WO_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Halton sequence for male position updates
double WO_halton_sequence(int index, int base) {
    double result = 0.0;
    double f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i = i / base;
        f /= base;
    }
    return result;
}

// Levy flight step for child position updates
void WO_levy_flight(double *step, int dim) {
    double beta = WO_LEVY_BETA;
    double sigma = pow((tgamma(1 + beta) * sin(M_PI * beta / 2) /
                       (tgamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))), 1 / beta);
    
    for (int i = 0; i < dim; i++) {
        double u = 0.0, v = 0.0;
        double r = WO_rand_double(0, 1);
        double s = WO_rand_double(0, 1);
        u = sigma * sqrt(-2.0 * log(r)) * cos(2.0 * M_PI * s);
        v = sqrt(-2.0 * log(r)) * sin(2.0 * M_PI * s);
        step[i] = u / pow(fabs(v), 1 / beta);
    }
}

// Migration Phase
void WO_migration_phase(Optimizer *opt, double beta, double r3) {
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    // Shuffle indices
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = j;
        indices[j] = temp;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += (beta * r3 * r3) * 
                                             (opt->population[indices[i]].position[j] - opt->population[i].position[j]);
        }
    }
    free(indices);
    enforce_bound_constraints(opt);
}

// Male Position Update (using Halton sequence)
void WO_male_position_update(Optimizer *opt) {
    int male_count = (int)(opt->population_size * WO_FEMALE_PROPORTION);
    for (int i = 0; i < male_count; i++) {
        double halton_val = WO_halton_sequence(i + 1, WO_HALTON_BASE);
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            halton_val * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Female Position Update
void WO_female_position_update(Optimizer *opt, double alpha) {
    int male_count = (int)(opt->population_size * WO_FEMALE_PROPORTION);
    int female_count = male_count;
    for (int i = male_count; i < male_count + female_count; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] += alpha * (opt->population[i - male_count].position[j] - 
                                                     opt->population[i].position[j]) +
                                             (1.0 - alpha) * (opt->best_solution.position[j] - 
                                                             opt->population[i].position[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Child Position Update (using Levy flight)
void WO_child_position_update(Optimizer *opt) {
    int male_count = (int)(opt->population_size * WO_FEMALE_PROPORTION);
    int female_count = male_count;
    int child_count = opt->population_size - male_count - female_count;
    double *levy_step = (double *)malloc(opt->dim * sizeof(double));
    double *o = (double *)malloc(opt->dim * sizeof(double));

    for (int i = opt->population_size - child_count; i < opt->population_size; i++) {
        double P = WO_rand_double(0, 1);
        WO_levy_flight(levy_step, opt->dim);
        for (int j = 0; j < opt->dim; j++) {
            o[j] = opt->best_solution.position[j] + opt->population[i].position[j] * levy_step[j];
            opt->population[i].position[j] = P * (o[j] - opt->population[i].position[j]);
        }
    }
    free(levy_step);
    free(o);
    enforce_bound_constraints(opt);
}

// Position Adjustment Phase
void WO_position_adjustment_phase(Optimizer *opt, double R) {
    for (int i = 0; i < opt->population_size; i++) {
        double r4 = WO_rand_double(0, 1);
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->population[i].position[j] * R - 
                                            fabs(opt->best_solution.position[j] - 
                                                opt->population[i].position[j]) * r4 * r4;
        }
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase (around best and second-best positions)
void WO_exploitation_phase(Optimizer *opt, double beta) {
    double *second_best = (double *)malloc(opt->dim * sizeof(double));
    double second_score = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness > opt->best_solution.fitness && 
            opt->population[i].fitness < second_score) {
            second_score = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                second_best[j] = opt->population[i].position[j];
            }
        }
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double theta1 = WO_rand_double(0, 1);
            double a1 = beta * WO_rand_double(0, 1) - beta;
            double b1 = tan(theta1 * M_PI);
            double X1 = opt->best_solution.position[j] - a1 * b1 * 
                       fabs(opt->best_solution.position[j] - opt->population[i].position[j]);

            double theta2 = WO_rand_double(0, 1);
            double a2 = beta * WO_rand_double(0, 1) - beta;
            double b2 = tan(theta2 * M_PI);
            double X2 = second_best[j] - a2 * b2 * 
                       fabs(second_best[j] - opt->population[i].position[j]);

            opt->population[i].position[j] = (X1 + X2) / 2.0;
        }
    }
    free(second_best);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    double best_score = INFINITY;
    double *second_best = (double *)malloc(opt->dim * sizeof(double));
    double second_score = INFINITY;

    for (int t = 0; t < opt->max_iter; t++) {
        // Evaluate fitness and update best and second-best positions
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < best_score) {
                best_score = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
                opt->best_solution.fitness = best_score;
            }
            if (best_score < opt->population[i].fitness && opt->population[i].fitness < second_score) {
                second_score = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    second_best[j] = opt->population[i].position[j];
                }
            }
        }

        // Update parameters
        double alpha = 1.0 - (double)t / opt->max_iter;
        double beta = 1.0 - 1.0 / (1.0 + exp((0.5 * opt->max_iter - t) / opt->max_iter * 10));
        double A = 2.0 * alpha;
        double r1 = WO_rand_double(0, 1);
        double R = 2.0 * r1 - 1.0;
        double danger_signal = A * R;
        double safety_signal = WO_rand_double(0, 1);

        if (fabs(danger_signal) >= 1.0) {
            double r3 = WO_rand_double(0, 1);
            WO_migration_phase(opt, beta, r3);
        } else {
            if (safety_signal >= 0.5) {
                WO_male_position_update(opt);
                WO_female_position_update(opt, alpha);
                WO_child_position_update(opt);
            } else if (safety_signal < 0.5 && fabs(danger_signal) >= 0.5) {
                WO_position_adjustment_phase(opt, R);
            } else {
                WO_exploitation_phase(opt, beta);
            }
        }
    }

    free(second_best);
}
