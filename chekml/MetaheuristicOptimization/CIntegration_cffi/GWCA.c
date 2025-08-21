#include "GWCA.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>  // for memcpy

// Faster rand() using static LCG (Linear Congruential Generator)
static inline double fast_rand(double max) {
    static unsigned int seed = 123456789;
    seed = (1103515245 * seed + 12345);
    return (double)(seed % 10000) / 10000.0 * max;
}

// Faster fitness comparator (inline-like behavior)
int compare_fitness_gwca(const void *a, const void *b) {
    double fa = ((Solution*)a)->fitness;
    double fb = ((Solution*)b)->fitness;
    return (fa > fb) - (fa < fb);
}

void GWCA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    double best_overall = INFINITY;
    Solution Worker1, Worker2, Worker3;

    qsort(opt->population, opt->population_size, sizeof(Solution), compare_fitness_gwca);
    Worker1 = opt->population[0];
    Worker2 = opt->population[1];
    Worker3 = opt->population[2];

    best_overall = Worker1.fitness;
    int LNP = (int)(opt->population_size * E + 0.999999);  // ceil without math.h

    double* W1 = Worker1.position;
    double* W2 = Worker2.position;
    double* W3 = Worker3.position;
    int dim = opt->dim;
    int pop_size = opt->population_size;

    for (int t = 1; t <= opt->max_iter; t++) {
        double C = CMAX - ((CMAX - CMIN) * t / opt->max_iter);
        double inv_iter_plus1 = 1.0 / (1.0 + t);
        double common_F = (M * G) / (P * Q);

        for (int i = 0; i < pop_size; i++) {
            double r1 = fast_rand(1.0);
            double r2 = fast_rand(1.0);
            double* pos = opt->population[i].position;

            if (i < LNP) {
                double F = common_F * r1 * inv_iter_plus1;
                double offset = C * F;
                for (int d = 0; d < dim; d++) {
                    double sign = (fast_rand(1.0) > 0.5) ? 1.0 : -1.0;
                    pos[d] += sign * offset;
                }
            } else {
                for (int d = 0; d < dim; d++) {
                    double avg = (W1[d] + W2[d] + W3[d]) * (1.0 / 3.0);
                    pos[d] += r2 * avg * C;
                }
            }

            enforce_bound_constraints(opt);
            double fit = objective_function(pos);
            opt->population[i].fitness = fit;

            if (fit < Worker1.fitness) {
                Worker3 = Worker2;
                Worker2 = Worker1;
                Worker1 = opt->population[i];

                W1 = Worker1.position;
                W2 = Worker2.position;
                W3 = Worker3.position;
            }
        }

        if (Worker1.fitness < best_overall) {
            best_overall = Worker1.fitness;
            memcpy(opt->best_solution.position, Worker1.position, sizeof(double) * dim);
            opt->best_solution.fitness = Worker1.fitness;
        }

        printf("Iteration %d: Best Fitness = %f\n", t, Worker1.fitness);
    }
}
