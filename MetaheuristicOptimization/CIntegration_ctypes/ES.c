#include "ES.h"
#include "generaloptimizer.h"

#include <float.h>
#include <time.h>

static double **velocity;
static double **local_best;
static double *local_best_cost;
static double *global_best_position;
static double global_best_cost;

// Helper function for uniform random double
double rand_double_es(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Initialization function
static void initialize(Optimizer *opt, double (*objective_function)(double *)) {
    int pop = opt->population_size;
    int dim = opt->dim;
    const double *bounds = opt->bounds;

    velocity = (double **)malloc(pop * sizeof(double *));
    local_best = (double **)malloc(pop * sizeof(double *));
    local_best_cost = (double *)malloc(pop * sizeof(double));
    global_best_position = (double *)malloc(dim * sizeof(double));

    for (int i = 0; i < pop; i++) {
        velocity[i] = (double *)malloc(dim * sizeof(double));
        local_best[i] = (double *)malloc(dim * sizeof(double));

        for (int j = 0; j < dim; j++) {
            velocity[i][j] = rand_double_es(0.0, 1.0);
            local_best[i][j] = opt->population[i].position[j];
        }

        local_best_cost[i] = objective_function(local_best[i]);
    }

    // Set global best
    int best_idx = 0;
    global_best_cost = local_best_cost[0];
    for (int i = 1; i < pop; i++) {
        if (local_best_cost[i] < global_best_cost) {
            global_best_cost = local_best_cost[i];
            best_idx = i;
        }
    }
    for (int j = 0; j < dim; j++) {
        global_best_position[j] = local_best[best_idx][j];
    }
}

static void update_velocity_and_position(Optimizer *opt, int iter) {
    int pop = opt->population_size;
    int dim = opt->dim;
    const double *bounds = opt->bounds;
    double w = W_MAX - ((W_MAX - W_MIN) * iter / (double)opt->max_iter);

    for (int i = 0; i < pop; i++) {
        for (int j = 0; j < dim; j++) {
            double r1 = rand_double_es(0.0, 1.0);
            double r2 = rand_double_es(0.0, 1.0);

            double cognitive = C1 * r1 * (local_best[i][j] - opt->population[i].position[j]);
            double social = C2 * r2 * (global_best_position[j] - opt->population[i].position[j]);

            velocity[i][j] = w * velocity[i][j] + cognitive + social;
            opt->population[i].position[j] += velocity[i][j];

            // Clamp
            if (opt->population[i].position[j] < bounds[2 * j])
                opt->population[i].position[j] = bounds[2 * j];
            if (opt->population[i].position[j] > bounds[2 * j + 1])
                opt->population[i].position[j] = bounds[2 * j + 1];
        }
    }
}

static void levy_flight(Optimizer *opt, double (*objective_function)(double *)) {
    int dim = opt->dim;
    int pop = opt->population_size;
    const double *bounds = opt->bounds;

    double beta = LEVY_BETA;
    double sigma = pow((tgamma(1 + beta) * sin(M_PI * beta / 2)) /
                       (tgamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2)), 1.0 / beta);

    for (int i = 0; i < pop; i++) {
        double s[dim];
        for (int j = 0; j < dim; j++) {
            double u = rand_double_es(0.0, 1.0) * sigma;
            double v = rand_double_es(0.0, 1.0);
            double step = u / pow(fabs(v), 1.0 / beta);

            s[j] = local_best[i][j] + LEVY_STEP_SCALE * step * (local_best[i][j] - global_best_position[j]);

            // Clamp
            if (s[j] < bounds[2 * j]) s[j] = bounds[2 * j];
            if (s[j] > bounds[2 * j + 1]) s[j] = bounds[2 * j + 1];
        }

        double s_cost = objective_function(s);
        if (s_cost < global_best_cost) {
            global_best_cost = s_cost;
            for (int j = 0; j < dim; j++) {
                global_best_position[j] = s[j];
            }
        }
    }
}

// Main optimization loop
void ES_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));
    initialize(opt, objective_function);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        if (rand_double_es(0.0, 1.0) < LEVY_PROBABILITY) {
            levy_flight(opt, objective_function);
            continue;
        }

        update_velocity_and_position(opt, iter);

        for (int i = 0; i < opt->population_size; i++) {
            double fitness = objective_function(opt->population[i].position);

            if (fitness < local_best_cost[i]) {
                for (int j = 0; j < opt->dim; j++) {
                    local_best[i][j] = opt->population[i].position[j];
                }
                local_best_cost[i] = fitness;
            }

            if (fitness < global_best_cost) {
                global_best_cost = fitness;
                for (int j = 0; j < opt->dim; j++) {
                    global_best_position[j] = opt->population[i].position[j];
                }
            }
        }
    }

    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = global_best_position[j];
    }
    opt->best_solution.fitness = global_best_cost;

    // Cleanup
    for (int i = 0; i < opt->population_size; i++) {
        free(velocity[i]);
        free(local_best[i]);
    }
    free(velocity);
    free(local_best);
    free(local_best_cost);
    free(global_best_position);
}

