#include "SWO.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

static inline double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static inline double levy_flight_swo() {
    const double sigma = 0.696066;
    double u = rand_double(-1.0, 1.0) * sigma;
    double v = rand_double(0.0001, 1.0); // Avoid zero
    return LEVY_SCALE * (u / pow(v, 0.666667)); // 1 / 1.5 = 0.666667
}

void SWO_hunting_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position || !opt->bounds) return;

    double new_pos[opt->dim];
    double bound_range[opt->dim];

    for (int j = 0; j < opt->dim; ++j)
        bound_range[j] = opt->bounds[2 * j + 1] - opt->bounds[2 * j];

    for (int i = 0; i < opt->population_size; ++i) {
        double *pos = opt->population[i].position;
        if (!pos) continue;

        double r1 = rand_double(0.0, 1.0);
        double r2 = rand_double(0.0, 1.0);
        double L = levy_flight_swo();

        for (int j = 0; j < opt->dim; ++j) {
            double val = (r1 < TRADE_OFF) ?
                pos[j] + L * (opt->best_solution.position[j] - pos[j]) :
                pos[j] + r2 * bound_range[j] * rand_double(-0.1, 0.1);
            new_pos[j] = fmin(fmax(val, opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }

        double new_fit = objective_function(new_pos);
        if (new_fit < opt->population[i].fitness) {
            opt->population[i].fitness = new_fit;
            for (int j = 0; j < opt->dim; ++j)
                pos[j] = new_pos[j];
            if (new_fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fit;
                for (int j = 0; j < opt->dim; ++j)
                    opt->best_solution.position[j] = new_pos[j];
            }
        }
    }
}

void SWO_mating_phase(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds) return;

    double new_pos[opt->dim];

    for (int i = 0; i < opt->population_size; ++i) {
        double *pos = opt->population[i].position;
        if (!pos) continue;

        int mate_idx = rand() % opt->population_size;
        if (mate_idx == i || !opt->population[mate_idx].position) continue;

        for (int j = 0; j < opt->dim; ++j) {
            double val = (rand_double(0.0, 1.0) < CROSSOVER_PROB) ?
                pos[j] + rand_double(0.0, 1.0) * (opt->population[mate_idx].position[j] - pos[j]) :
                pos[j];
            new_pos[j] = fmin(fmax(val, opt->bounds[2 * j]), opt->bounds[2 * j + 1]);
        }

        double new_fit = objective_function(new_pos);
        if (new_fit < opt->population[i].fitness) {
            opt->population[i].fitness = new_fit;
            for (int j = 0; j < opt->dim; ++j)
                pos[j] = new_pos[j];
            if (new_fit < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fit;
                for (int j = 0; j < opt->dim; ++j)
                    opt->best_solution.position[j] = new_pos[j];
            }
        }
    }
}

void SWO_population_reduction(Optimizer *opt, int iter) {
    if (!opt || !opt->population || !opt->population[0].position) return;

    int new_size = MIN_POPULATION + (int)((opt->population_size - MIN_POPULATION) * ((double)(opt->max_iter - iter) / opt->max_iter));
    if (new_size < MIN_POPULATION) new_size = MIN_POPULATION;
    if (new_size >= opt->population_size) return;

    // Insertion sort
    for (int i = 1; i < opt->population_size; ++i) {
        Solution key = opt->population[i];
        int j = i - 1;
        while (j >= 0 && opt->population[j].fitness > key.fitness) {
            opt->population[j + 1] = opt->population[j];
            --j;
        }
        opt->population[j + 1] = key;
    }

    opt->population_size = new_size;
}

void SWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->best_solution.position) return;

    srand((unsigned int)time(NULL));

    for (int i = 0; i < opt->population_size; ++i) {
        double fit = objective_function(opt->population[i].position);
        opt->population[i].fitness = fit;
        if (fit < opt->best_solution.fitness) {
            opt->best_solution.fitness = fit;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }

    for (int iter = 0; iter < opt->max_iter; ++iter) {
        if (rand_double(0.0, 1.0) < TRADE_OFF)
            SWO_hunting_phase(opt, objective_function);
        else
            SWO_mating_phase(opt, objective_function);
        SWO_population_reduction(opt, iter);
    }
}

