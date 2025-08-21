#include "SA.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// 🔧 Generate a neighbor by perturbing a random dimension slightly
static void generate_neighbor(const double* x, double* out, int dim, double* bounds) {
    memcpy(out, x, sizeof(double) * dim);
    int idx = rand() % dim;
    double perturbation = ((double)rand() / RAND_MAX - 0.5) / 50.0;  // ~randn / 100
    out[idx] += perturbation;

    // Enforce bounds
    double min = bounds[2 * idx];
    double max = bounds[2 * idx + 1];
    if (out[idx] < min) out[idx] = min;
    if (out[idx] > max) out[idx] = max;
}

// ❄️ Cooling schedule
static double cool(double T) {
    return SA_COOLING_RATE * T;
}

// 🚀 Simulated Annealing Optimizer
void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt) return;

    int dim = opt->dim;
    double* bounds = opt->bounds;

    // Allocate memory
    double* parent = (double*)malloc(sizeof(double) * dim);
    double* current = (double*)malloc(sizeof(double) * dim);
    double* newsol = (double*)malloc(sizeof(double) * dim);

    // Initialize parent randomly
    for (int d = 0; d < dim; d++) {
        double min = bounds[2 * d];
        double max = bounds[2 * d + 1];
        parent[d] = min + ((double)rand() / RAND_MAX) * (max - min);
    }

    double T = SA_INIT_TEMP;
    double old_energy = objective_function(parent);
    double init_energy = old_energy;

    int itry = 0, success = 0, consec = 0, finished = 0, total = 0;
    double k = 1.0;

    if (SA_VERBOSITY == 2)
        printf("  T = %7.5f, loss = %10.5f\n", T, old_energy);

    while (!finished) {
        itry++;
        memcpy(current, parent, sizeof(double) * dim);

        // Stop condition
        if (itry >= SA_MAX_TRIES || success >= SA_MAX_SUCCESS) {
            if (T < SA_STOP_TEMP || consec >= SA_MAX_CONS_REJ) {
                finished = 1;
                total += itry;
                break;
            } else {
                T = cool(T);
                if (SA_VERBOSITY == 2)
                    printf("  T = %7.5f, loss = %10.5f\n", T, old_energy);
                total += itry;
                itry = 0;
                success = 0;
            }
        }

        // Generate neighbor and evaluate
        generate_neighbor(current, newsol, dim, bounds);
        double new_energy = objective_function(newsol);

        if (new_energy < SA_STOP_VAL) {
            memcpy(parent, newsol, sizeof(double) * dim);
            old_energy = new_energy;
            break;
        }

        double delta = old_energy - new_energy;

        if (delta > 1e-6) {
            memcpy(parent, newsol, sizeof(double) * dim);
            old_energy = new_energy;
            success++;
            consec = 0;
        } else {
            if (((double)rand() / RAND_MAX) < exp(delta / (k * T))) {
                memcpy(parent, newsol, sizeof(double) * dim);
                old_energy = new_energy;
                success++;
            } else {
                consec++;
            }
        }
    }

    // Store result in best_solution
    memcpy(opt->best_solution.position, parent, sizeof(double) * dim);
    opt->best_solution.fitness = old_energy;

    if (SA_VERBOSITY >= 1) {
        printf("Final Fitness = %f after %d iterations\n", old_energy, total + itry);
    }

    // Free memory
    free(parent);
    free(current);
    free(newsol);
}

