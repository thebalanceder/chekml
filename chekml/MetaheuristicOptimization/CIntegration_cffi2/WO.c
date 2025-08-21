#include "WO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Levy flight step for child position updates
void WO_levy_flight(double *step, int dim, double sigma) {
    for (int i = 0; i < dim; i++) {
        double r = WO_rand_double(0, 1);
        double s = WO_rand_double(0, 1);
        double u = sigma * sqrt(-2.0 * log(r)) * cos(2.0 * M_PI * s);
        double v = sqrt(-2.0 * log(r)) * sin(2.0 * M_PI * s);
        step[i] = u / pow(fabs(v), 1.0 / WO_LEVY_BETA);
    }
}

// Migration Phase
void WO_migration_phase(Optimizer *opt, WOContext *ctx, double beta, double r3) {
    int *indices = ctx->temp_indices;
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = (int)(WO_rand_double(0, i + 1));
        int temp = indices[i];
        indices[i] = j;
        indices[j] = temp;
    }

    double step = beta * r3 * r3;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double *other_pos = opt->population[indices[i]].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] += step * (other_pos[j] - pos[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Male Position Update
void WO_male_position_update(Optimizer *opt, WOContext *ctx) {
    int male_count = ctx->male_count;
    for (int i = 0; i < male_count; i++) {
        double halton_val = WO_halton_sequence(i + 1, WO_HALTON_BASE);
        double *pos = opt->population[i].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2 * j] + halton_val * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Female Position Update
void WO_female_position_update(Optimizer *opt, WOContext *ctx, double alpha) {
    int male_count = ctx->male_count;
    int female_count = ctx->female_count;
    double one_minus_alpha = 1.0 - alpha;
    for (int i = male_count; i < male_count + female_count; i++) {
        double *pos = opt->population[i].position;
        double *male_pos = opt->population[i - male_count].position;
        double *best_pos = opt->best_solution.position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] += alpha * (male_pos[j] - pos[j]) + one_minus_alpha * (best_pos[j] - pos[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Child Position Update
void WO_child_position_update(Optimizer *opt, WOContext *ctx) {
    int child_count = ctx->child_count;
    int start_idx = opt->population_size - child_count;
    double *levy_step = ctx->temp_array1;
    double *o = ctx->temp_array2;

    for (int i = start_idx; i < opt->population_size; i++) {
        double P = WO_rand_double(0, 1);
        WO_levy_flight(levy_step, opt->dim, ctx->levy_sigma);
        double *pos = opt->population[i].position;
        double *best_pos = opt->best_solution.position;
        for (int j = 0; j < opt->dim; j++) {
            o[j] = best_pos[j] + pos[j] * levy_step[j];
            pos[j] = P * (o[j] - pos[j]);
        }
    }
    enforce_bound_constraints(opt);
}

// Position Adjustment Phase
void WO_position_adjustment_phase(Optimizer *opt, WOContext *ctx, double R) {
    for (int i = 0; i < opt->population_size; i++) {
        double r4 = WO_rand_double(0, 1);
        double r4_squared = r4 * r4;
        double *pos = opt->population[i].position;
        double *best_pos = opt->best_solution.position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = pos[j] * R - fabs(best_pos[j] - pos[j]) * r4_squared;
        }
    }
    enforce_bound_constraints(opt);
}

// Exploitation Phase
void WO_exploitation_phase(Optimizer *opt, WOContext *ctx, double beta) {
    double *second_best = ctx->second_best.position;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = opt->population[i].position;
        double *best_pos = opt->best_solution.position;
        for (int j = 0; j < opt->dim; j++) {
            double theta1 = WO_rand_double(0, 1);
            double a1 = beta * WO_rand_double(0, 1) - beta;
            double b1 = tan(theta1 * M_PI);
            double X1 = best_pos[j] - a1 * b1 * fabs(best_pos[j] - pos[j]);

            double theta2 = WO_rand_double(0, 1);
            double a2 = beta * WO_rand_double(0, 1) - beta;
            double b2 = tan(theta2 * M_PI);
            double X2 = second_best[j] - a2 * b2 * fabs(second_best[j] - pos[j]);

            pos[j] = (X1 + X2) / 2.0;
        }
    }
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize WOContext
    WOContext ctx = {0};
    ctx.male_count = (int)(opt->population_size * WO_FEMALE_PROPORTION);
    ctx.female_count = ctx.male_count;
    ctx.child_count = opt->population_size - ctx.male_count - ctx.female_count;
    double levy_beta = WO_LEVY_BETA;
    ctx.levy_sigma = pow((tgamma(1 + levy_beta) * sin(M_PI * levy_beta / 2) /
                         (tgamma((1 + levy_beta) / 2) * levy_beta * pow(2, (levy_beta - 1) / 2))), 1 / levy_beta);
    
    ctx.temp_indices = (int *)malloc(opt->population_size * sizeof(int));
    ctx.temp_array1 = (double *)malloc(opt->dim * sizeof(double));
    ctx.temp_array2 = (double *)malloc(opt->dim * sizeof(double));
    ctx.second_best.position = (double *)malloc(opt->dim * sizeof(double));
    ctx.second_best.fitness = INFINITY;

    // Initial fitness evaluation
    #pragma omp parallel for if(opt->population_size > 100)
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    double best_score = INFINITY;
    double second_score = INFINITY;
    for (int t = 0; t < opt->max_iter; t++) {
        // Update best and second-best solutions
        #pragma omp parallel for if(opt->population_size > 100)
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = opt->population[i].fitness;
            if (fitness < best_score) {
                #pragma omp critical
                {
                    if (fitness < best_score) {
                        best_score = fitness;
                        for (int j = 0; j < opt->dim; j++) {
                            opt->best_solution.position[j] = opt->population[i].position[j];
                        }
                        opt->best_solution.fitness = best_score;
                    }
                }
            }
            if (best_score < fitness && fitness < second_score) {
                #pragma omp critical
                {
                    if (best_score < fitness && fitness < second_score) {
                        second_score = fitness;
                        for (int j = 0; j < opt->dim; j++) {
                            ctx.second_best.position[j] = opt->population[i].position[j];
                        }
                        ctx.second_best.fitness = second_score;
                    }
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
            WO_migration_phase(opt, &ctx, beta, r3);
        } else if (safety_signal >= 0.5) {
            WO_male_position_update(opt, &ctx);
            WO_female_position_update(opt, &ctx, alpha);
            WO_child_position_update(opt, &ctx);
        } else if (fabs(danger_signal) >= 0.5) {
            WO_position_adjustment_phase(opt, &ctx, R);
        } else {
            WO_exploitation_phase(opt, &ctx, beta);
        }

        // Re-evaluate fitness
        #pragma omp parallel for if(opt->population_size > 100)
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
    }

    // Free WOContext resources
    free(ctx.temp_indices);
    free(ctx.temp_array1);
    free(ctx.temp_array2);
    free(ctx.second_best.position);
}
