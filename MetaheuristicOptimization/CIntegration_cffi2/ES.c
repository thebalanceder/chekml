#include "ES.h"
#include "generaloptimizer.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double **velocity;
static double **local_best;
static double *local_best_cost;
static double *global_best_position;
static double global_best_cost;

#define RAND_DOUBLE_ES(min, max) ((min) + ((double)rand() / RAND_MAX) * ((max) - (min)))

static void initialize(Optimizer *opt, double (*objective_function)(double *)) {
    int pop = opt->population_size;
    int dim = opt->dim;
    const double *bounds = opt->bounds;

    velocity = (double **)malloc(pop * sizeof(double *));
    local_best = (double **)malloc(pop * sizeof(double *));
    local_best_cost = (double *)malloc(pop * sizeof(double));
    global_best_position = (double *)malloc(dim * sizeof(double));

    int best_idx = 0;
    global_best_cost = DBL_MAX;

    for (int i = 0; i < pop; i++) {
        velocity[i] = (double *)malloc(dim * sizeof(double));
        local_best[i] = (double *)malloc(dim * sizeof(double));

        for (int j = 0; j < dim; j++) {
            velocity[i][j] = RAND_DOUBLE_ES(-1.0, 1.0);  // symmetric velocity
            local_best[i][j] = opt->population[i].position[j];
        }

        local_best_cost[i] = objective_function(local_best[i]);

        if (local_best_cost[i] < global_best_cost) {
            global_best_cost = local_best_cost[i];
            best_idx = i;
        }
    }

    memcpy(global_best_position, local_best[best_idx], dim * sizeof(double));
}

static void update_velocity_and_position(Optimizer *restrict opt, int iter) {
    int pop = opt->population_size;
    int dim = opt->dim;
    const double *restrict bounds = opt->bounds;
    double w = W_MAX - ((W_MAX - W_MIN) * iter / (double)opt->max_iter);

    for (int i = 0; i < pop; i++) {
        double *restrict pos = opt->population[i].position;
        double *restrict v = velocity[i];
        double *restrict pbest = local_best[i];

        for (int j = 0; j < dim; j++) {
            double r1 = RAND_DOUBLE_ES(0.0, 1.0);
            double r2 = RAND_DOUBLE_ES(0.0, 1.0);
            double cog = C1 * r1 * (pbest[j] - pos[j]);
            double soc = C2 * r2 * (global_best_position[j] - pos[j]);

            v[j] = w * v[j] + cog + soc;
            pos[j] += v[j];

            // Branchless clamp
            pos[j] = fmax(bounds[2 * j], fmin(pos[j], bounds[2 * j + 1]));
        }
    }
}

static void levy_flight(Optimizer *opt, double (*objective_function)(double *)) {
    int pop = opt->population_size;
    int dim = opt->dim;
    const double *bounds = opt->bounds;

    static const double beta = LEVY_BETA;
    static const double sigma = 
        pow((tgamma(1 + beta) * sin(M_PI * beta / 2)) /
            (tgamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2)), 1.0 / beta);

    for (int i = 0; i < pop; i++) {
        double *pbest = local_best[i];
        double s[dim];

        for (int j = 0; j < dim; j++) {
            double u = RAND_DOUBLE_ES(0.0, 1.0) * sigma;
            double v = RAND_DOUBLE_ES(0.0, 1.0);
            double step = u / pow(fabs(v), 1.0 / beta);

            s[j] = pbest[j] + LEVY_STEP_SCALE * step * (pbest[j] - global_best_position[j]);

            // Clamp
            if (s[j] < bounds[2 * j]) s[j] = bounds[2 * j];
            else if (s[j] > bounds[2 * j + 1]) s[j] = bounds[2 * j + 1];
        }

        double s_cost = objective_function(s);
        if (s_cost < global_best_cost) {
            global_best_cost = s_cost;
            memcpy(global_best_position, s, dim * sizeof(double));
        }
    }
}

void ES_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL));
    initialize(opt, objective_function);

    int pop = opt->population_size;
    int dim = opt->dim;

    for (int iter = 0; iter < opt->max_iter; iter++) {
        if (RAND_DOUBLE_ES(0.0, 1.0) < LEVY_PROBABILITY) {
            levy_flight(opt, objective_function);
        } else {
            update_velocity_and_position(opt, iter);

            for (int i = 0; i < pop; i++) {
                double *pos = opt->population[i].position;
                double fit = objective_function(pos);

                if (fit < local_best_cost[i]) {
                    memcpy(local_best[i], pos, dim * sizeof(double));
                    local_best_cost[i] = fit;
                }

                if (fit < global_best_cost) {
                    global_best_cost = fit;
                    memcpy(global_best_position, pos, dim * sizeof(double));
                }
            }
        }
    }

    // Save final best
    memcpy(opt->best_solution.position, global_best_position, dim * sizeof(double));
    opt->best_solution.fitness = global_best_cost;

    // Cleanup
    for (int i = 0; i < pop; i++) {
        free(velocity[i]);
        free(local_best[i]);
    }
    free(velocity);
    free(local_best);
    free(local_best_cost);
    free(global_best_position);
}
