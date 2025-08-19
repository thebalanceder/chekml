#include "WO.h"
#include "generaloptimizer.h"
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

// Fill random number buffer
static inline void fill_rand_buffer(WOContext *ctx) {
    for (int i = 0; i < WO_RAND_BUFFER_SIZE; i++) {
        ctx->rng.state ^= ctx->rng.state >> 12;
        ctx->rng.state ^= ctx->rng.state << 25;
        ctx->rng.state ^= ctx->rng.state >> 27;
        uint64_t tmp = ctx->rng.state * 2685821657736338717ULL;
        ctx->rand_buffer[i] = (double)(tmp >> 32) / UINT32_MAX;
    }
    ctx->rand_index = 0;
}

// Get random double from buffer
static inline double get_rand(WOContext *ctx, double min, double max) {
    if (ctx->rand_index >= WO_RAND_BUFFER_SIZE) {
        fill_rand_buffer(ctx);
    }
    return min + (max - min) * ctx->rand_buffer[ctx->rand_index++];
}

// Main Optimization Function
void WO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Check constraints
    if (opt->dim > WO_MAX_DIM || opt->population_size > WO_MAX_POP) {
        fprintf(stderr, "Error: dim (%d) or population_size (%d) exceeds maximum (%d, %d)\n",
                opt->dim, opt->population_size, WO_MAX_DIM, WO_MAX_POP);
        return;
    }

    // Initialize WOContext
    WOContext ctx = {0};
    ctx.male_count = (int)(opt->population_size * WO_FEMALE_PROPORTION);
    ctx.female_count = ctx.male_count;
    ctx.child_count = opt->population_size - ctx.male_count - ctx.female_count;
    double levy_beta = WO_LEVY_BETA;
    ctx.levy_sigma = pow((tgamma(1 + levy_beta) * sin(M_PI * levy_beta / 2) /
                         (tgamma((1 + levy_beta) / 2) * levy_beta * pow(2, (levy_beta - 1) / 2))), 1 / levy_beta);
    ctx.second_best.position = ctx.second_best_pos;
    ctx.second_best.fitness = INFINITY;
    ctx.rng.state = (uint64_t)time(NULL) ? (uint64_t)time(NULL) : 88172645463325252ULL;
    fill_rand_buffer(&ctx);

    // Initial fitness evaluation
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    double best_score = INFINITY;
    double second_score = INFINITY;
    for (int t = 0; t < opt->max_iter; t++) {
        // Update best and second-best solutions
        for (int i = 0; i < opt->population_size; i++) {
            double fitness = opt->population[i].fitness;
            if (fitness < best_score) {
                best_score = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
                opt->best_solution.fitness = best_score;
            }
            if (best_score < fitness && fitness < second_score) {
                second_score = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    ctx.second_best.position[j] = opt->population[i].position[j];
                }
                ctx.second_best.fitness = second_score;
            }
        }

        // Update parameters
        double alpha = 1.0 - (double)t / opt->max_iter;
        double beta = 1.0 - 1.0 / (1.0 + exp((0.5 * opt->max_iter - t) / opt->max_iter * 10));
        double A = 2.0 * alpha;
        double r1 = get_rand(&ctx, 0, 1);
        double R = 2.0 * r1 - 1.0;
        double danger_signal = A * R;
        double safety_signal = get_rand(&ctx, 0, 1);

        if (fabs(danger_signal) >= 1.0) {
            // Migration Phase
            double r3 = get_rand(&ctx, 0, 1);
            int *indices = ctx.temp_indices;
            for (int i = 0; i < opt->population_size; i++) {
                indices[i] = i;
            }
            for (int i = opt->population_size - 1; i > 0; i--) {
                int j = (int)(get_rand(&ctx, 0, i + 1));
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
        } else if (safety_signal >= 0.5) {
            // Male Position Update
            for (int i = 0; i < ctx.male_count; i++) {
                double result = 0.0;
                double f = 1.0 / WO_HALTON_BASE;
                int idx = i + 1;
                while (idx > 0) {
                    result += f * (idx % WO_HALTON_BASE);
                    idx = idx / WO_HALTON_BASE;
                    f /= WO_HALTON_BASE;
                }
                double *pos = opt->population[i].position;
                for (int j = 0; j < opt->dim; j++) {
                    pos[j] = opt->bounds[2 * j] + result * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
                }
            }
            enforce_bound_constraints(opt);

            // Female Position Update
            double one_minus_alpha = 1.0 - alpha;
            for (int i = ctx.male_count; i < ctx.male_count + ctx.female_count; i++) {
                double *pos = opt->population[i].position;
                double *male_pos = opt->population[i - ctx.male_count].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < opt->dim; j++) {
                    pos[j] += alpha * (male_pos[j] - pos[j]) + one_minus_alpha * (best_pos[j] - pos[j]);
                }
            }
            enforce_bound_constraints(opt);

            // Child Position Update
            int start_idx = opt->population_size - ctx.child_count;
            double *levy_step = ctx.temp_array1;
            double *o = ctx.temp_array2;
            for (int i = start_idx; i < opt->population_size; i++) {
                double P = get_rand(&ctx, 0, 1);
                for (int j = 0; j < opt->dim; j++) {
                    double r = get_rand(&ctx, 0, 1);
                    double s = get_rand(&ctx, 0, 1);
                    double u = ctx.levy_sigma * sqrt(-2.0 * log(r)) * cos(2.0 * M_PI * s);
                    double v = sqrt(-2.0 * log(r)) * sin(2.0 * M_PI * s);
                    levy_step[j] = u / pow(fabs(v), 1.0 / WO_LEVY_BETA);
                }
                double *pos = opt->population[i].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < opt->dim; j++) {
                    o[j] = best_pos[j] + pos[j] * levy_step[j];
                    pos[j] = P * (o[j] - pos[j]);
                }
            }
            enforce_bound_constraints(opt);
        } else if (fabs(danger_signal) >= 0.5) {
            // Position Adjustment Phase
            for (int i = 0; i < opt->population_size; i++) {
                double r4 = get_rand(&ctx, 0, 1);
                double r4_squared = r4 * r4;
                double *pos = opt->population[i].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < opt->dim; j++) {
                    pos[j] = pos[j] * R - fabs(best_pos[j] - pos[j]) * r4_squared;
                }
            }
            enforce_bound_constraints(opt);
        } else {
            // Exploitation Phase
            double *second_best = ctx.second_best.position;
            for (int i = 0; i < opt->population_size; i++) {
                double *pos = opt->population[i].position;
                double *best_pos = opt->best_solution.position;
                for (int j = 0; j < opt->dim; j++) {
                    double theta1 = get_rand(&ctx, 0, 1);
                    double a1 = beta * get_rand(&ctx, 0, 1) - beta;
                    double b1 = tan(theta1 * M_PI);
                    double X1 = best_pos[j] - a1 * b1 * fabs(best_pos[j] - pos[j]);

                    double theta2 = get_rand(&ctx, 0, 1);
                    double a2 = beta * get_rand(&ctx, 0, 1) - beta;
                    double b2 = tan(theta2 * M_PI);
                    double X2 = second_best[j] - a2 * b2 * fabs(second_best[j] - pos[j]);

                    pos[j] = (X1 + X2) / 2.0;
                }
            }
            enforce_bound_constraints(opt);
        }

        // Re-evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
        }
    }
}
