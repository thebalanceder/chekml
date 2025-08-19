#include "SA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

static inline double random_uniform(double min, double max) {
    return min + ((double)rand() / (double)RAND_MAX) * (max - min);
}

// 🧪 Small Gaussian perturbation on one dimension
static inline void perturb_solution(double* out, const double* in, int dim, const double* bounds) {
    memcpy(out, in, sizeof(double) * dim);
    int idx = rand() % dim;
    double perturb = ((double)rand() / RAND_MAX - 0.5) / 50.0;
    out[idx] += perturb;

    // Clamp bounds
    double min = bounds[2 * idx];
    double max = bounds[2 * idx + 1];
    if (out[idx] < min) out[idx] = min;
    if (out[idx] > max) out[idx] = max;
}

void SA_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) return;

    int dim = opt->dim;
    int max_iter = opt->max_iter;

    double* parent = (double*)malloc(sizeof(double) * dim);
    double* candidate = (double*)malloc(sizeof(double) * dim);
    double* bounds = opt->bounds;

    double T = INITIAL_TEMP;
    double old_energy, new_energy, delta;

    // 🎯 Initialize solution
    for (int d = 0; d < dim; ++d) {
        double min = bounds[2 * d];
        double max = bounds[2 * d + 1];
        parent[d] = random_uniform(min, max);
    }

    old_energy = objective_function(parent);
    memcpy(opt->best_solution.position, parent, sizeof(double) * dim);
    opt->best_solution.fitness = old_energy;

    int itry = 0, success = 0, consec_rej = 0, total_iter = 0;

    while (1) {
        ++itry;

        // 🔁 Cooling or stopping
        if (itry >= MAX_TRIES || success >= MAX_SUCCESS) {
            if (T < STOP_TEMP || consec_rej >= MAX_CONSEC_REJ) {
                break;
            } else {
                T *= COOLING_FACTOR;
                itry = 0;
                success = 0;
            }
        }

        perturb_solution(candidate, parent, dim, bounds);
        new_energy = objective_function(candidate);

        if (new_energy < -INFINITY){
            memcpy(parent, candidate, sizeof(double) * dim);
            old_energy = new_energy;
            break;
        }

        delta = old_energy - new_energy;

        if (delta > MIN_DELTA) {
            memcpy(parent, candidate, sizeof(double) * dim);
            old_energy = new_energy;
            success++;
            consec_rej = 0;
        } else if ((double)rand() / RAND_MAX < exp(delta / (BOLTZMANN_CONST * T))) {
            memcpy(parent, candidate, sizeof(double) * dim);
            old_energy = new_energy;
            success++;
            consec_rej = 0;
        } else {
            consec_rej++;
        }

        total_iter++;
        if (new_energy < opt->best_solution.fitness) {
            memcpy(opt->best_solution.position, candidate, sizeof(double) * dim);
            opt->best_solution.fitness = new_energy;
        }

        enforce_bound_constraints(opt); // Optional safeguard
    }

    // Store final solution
    memcpy(opt->best_solution.position, parent, sizeof(double) * dim);
    opt->best_solution.fitness = old_energy;

    free(parent);
    free(candidate);
}


