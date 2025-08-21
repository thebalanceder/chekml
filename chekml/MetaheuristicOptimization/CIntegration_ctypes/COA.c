#include "COA.h"
#include "generaloptimizer.h"
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time() if seeding random generator
#include <stdio.h>   // For debugging

// Function to generate a random double between min and max
double rand_double(double min, double max);
// Helper function for qsort
int compare_doubles_coa(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Initialize coyote population
void initialize_coyotes(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->best_solution.position || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer or memory allocation\n");
        return;
    }
    srand(time(NULL));  // Seed random number generator
    int total_coyotes = N_PACKS * N_COYOTES_PER_PACK;
    if (opt->population_size < total_coyotes || opt->dim <= 0) {
        fprintf(stderr, "Error: Invalid population_size (%d) or dim (%d), expected at least %d individuals\n", opt->population_size, opt->dim, total_coyotes);
        return;
    }
    int nfeval = 0;

    for (int c = 0; c < total_coyotes && c < opt->population_size; c++) {
        if (!opt->population[c].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", c);
            return;
        }
        for (int j = 0; j < opt->dim; j++) {
            opt->population[c].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[c].fitness = objective_function(opt->population[c].position);
        nfeval++;
    }

    int ibest = 0;
    for (int i = 1; i < total_coyotes && i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[ibest].fitness) {
            ibest = i;
        }
    }
    opt->best_solution.fitness = opt->population[ibest].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[ibest].position[j];
    }
}

// Update a single pack
void update_pack(Optimizer *opt, int pack_idx, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer or memory allocation\n");
        return;
    }
    int start_idx = pack_idx * N_COYOTES_PER_PACK;
    if (start_idx + N_COYOTES_PER_PACK > opt->population_size || opt->dim <= 0) {
        fprintf(stderr, "Error: Invalid pack_idx (%d) or dim (%d), start_idx (%d) exceeds population_size (%d)\n", pack_idx, opt->dim, start_idx, opt->population_size);
        return;
    }
    double pack_costs[N_COYOTES_PER_PACK];
    double pack_ages[N_COYOTES_PER_PACK];  // Simulate ages
    double new_coyotes[N_COYOTES_PER_PACK][opt->dim];
    double c_alpha[opt->dim];
    double tendency[opt->dim];
    int nfeval = 0;

    // Extract pack data
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        if (!opt->population[start_idx + i].position) {
            fprintf(stderr, "Error: population[%d].position is NULL\n", start_idx + i);
            return;
        }
        pack_costs[i] = opt->population[start_idx + i].fitness;
        pack_ages[i] = i + 1;  // Simulate age as index (simple placeholder)
    }

    // Sort pack by fitness
    int ind[N_COYOTES_PER_PACK];
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) ind[i] = i;
    for (int i = 0; i < N_COYOTES_PER_PACK - 1; i++) {
        for (int j = i + 1; j < N_COYOTES_PER_PACK; j++) {
            if (pack_costs[ind[i]] > pack_costs[ind[j]]) {
                int temp = ind[i];
                ind[i] = ind[j];
                ind[j] = temp;
            }
        }
    }

    // Reorder pack data
    double temp_costs[N_COYOTES_PER_PACK];
    double temp_ages[N_COYOTES_PER_PACK];
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        temp_costs[i] = pack_costs[ind[i]];
        temp_ages[i] = pack_ages[ind[i]];
    }
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        pack_costs[i] = temp_costs[i];
        pack_ages[i] = temp_ages[i];
    }
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[start_idx + i].position[j] = opt->population[start_idx + ind[i]].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        c_alpha[j] = opt->population[start_idx].position[j];
    }

    // Calculate tendency (median)
    for (int j = 0; j < opt->dim; j++) {
        double temp[N_COYOTES_PER_PACK];
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            temp[i] = opt->population[start_idx + i].position[j];
        }
        qsort(temp, N_COYOTES_PER_PACK, sizeof(double), compare_doubles_coa);
        tendency[j] = (N_COYOTES_PER_PACK % 2 == 0) ? (temp[N_COYOTES_PER_PACK / 2 - 1] + temp[N_COYOTES_PER_PACK / 2]) / 2.0 : temp[N_COYOTES_PER_PACK / 2];
    }

    // Diversity and scale factor
    double diversity = 0.0;
    for (int j = 0; j < opt->dim; j++) {
        double mean = 0.0;
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            mean += opt->population[start_idx + i].position[j];
        }
        mean /= N_COYOTES_PER_PACK;
        double std = 0.0;
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            std += pow(opt->population[start_idx + i].position[j] - mean, 2);
        }
        std = sqrt(std / N_COYOTES_PER_PACK);
        diversity += std;
    }
    diversity /= opt->dim;
    double scale_factor = (diversity < SCALE_FACTOR_DIVERSITY_THRESHOLD) ? SCALE_FACTOR_HIGH : SCALE_FACTOR_LOW;

    // Update coyotes
    for (int c = 0; c < N_COYOTES_PER_PACK; c++) {
        int rc1 = c;
        while (rc1 == c) rc1 = rand() % N_COYOTES_PER_PACK;
        int rc2 = c;
        while (rc2 == c || rc2 == rc1) rc2 = rand() % N_COYOTES_PER_PACK;

        for (int j = 0; j < opt->dim; j++) {
            new_coyotes[c][j] = opt->population[start_idx + c].position[j] +
                                scale_factor * rand_double(0.0, 1.0) * (c_alpha[j] - opt->population[start_idx + rc1].position[j]) +
                                scale_factor * rand_double(0.0, 1.0) * (tendency[j] - opt->population[start_idx + rc2].position[j]);
        }
        limit_bounds(opt, new_coyotes[c], opt->population[start_idx + c].position, opt->dim);

        double new_cost = objective_function(opt->population[start_idx + c].position);
        nfeval++;
        if (new_cost < pack_costs[c]) {
            pack_costs[c] = new_cost;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[start_idx + c].position[j] = new_coyotes[c][j];
            }
        }
    }

    // Breeding (simplified)
    int parents[2];
    for (int i = 0; i < 2; i++) parents[i] = rand() % N_COYOTES_PER_PACK;
    double pup[opt->dim];
    for (int j = 0; j < opt->dim; j++) {
        double r = rand_double(0.0, 1.0);
        if (r < PS / 2) pup[j] = opt->population[start_idx + parents[0]].position[j];
        else if (r < 1.0 - PS / 2) pup[j] = opt->population[start_idx + parents[1]].position[j];
        else pup[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
    }
    limit_bounds(opt, pup, pup, opt->dim);

    double pup_cost = objective_function(pup);
    nfeval++;
    int worst_idx = 0;
    for (int i = 1; i < N_COYOTES_PER_PACK; i++) {
        if (pack_costs[i] > pack_costs[worst_idx]) worst_idx = i;
    }
    if (pup_cost < pack_costs[worst_idx]) {
        pack_costs[worst_idx] = pup_cost;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[start_idx + worst_idx].position[j] = pup[j];
        }
        // Simulate age reset (no age member, so no action here)
    }

    // Update costs (no age update due to missing age member)
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        opt->population[start_idx + i].fitness = pack_costs[i];
    }
}

// Exchange coyotes between packs
void pack_exchange(Optimizer *opt) {
    if (!opt || !opt->population || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer or memory allocation\n");
        return;
    }
    int max_packs = opt->population_size / N_COYOTES_PER_PACK;
    if (N_PACKS > max_packs) {
        fprintf(stderr, "Error: N_PACKS (%d) exceeds maximum packs (%d) for population_size (%d)\n", N_PACKS, max_packs, opt->population_size);
        return;
    }
    if (N_PACKS > 1 && rand_double(0.0, 1.0) < P_LEAVE) {
        int rp[2];
        rp[0] = rand() % N_PACKS;
        rp[1] = rand() % N_PACKS;
        while (rp[1] == rp[0]) rp[1] = rand() % N_PACKS;
        int rc[2];
        rc[0] = rand() % N_COYOTES_PER_PACK;
        rc[1] = rand() % N_COYOTES_PER_PACK;
        int idx1 = rp[0] * N_COYOTES_PER_PACK + rc[0];
        int idx2 = rp[1] * N_COYOTES_PER_PACK + rc[1];
        if (idx1 >= opt->population_size || idx2 >= opt->population_size) {
            fprintf(stderr, "Error: Pack exchange indices (%d, %d) exceed population_size (%d)\n", idx1, idx2, opt->population_size);
            return;
        }
        double temp[opt->dim];
        for (int j = 0; j < opt->dim; j++) {
            temp[j] = opt->population[idx1].position[j];
            opt->population[idx1].position[j] = opt->population[idx2].position[j];
            opt->population[idx2].position[j] = temp[j];
        }
    }
}

// Limit bounds for a solution
void limit_bounds(Optimizer *opt, double *X, double *X_clipped, int size) {
    if (!opt || !opt->bounds || !X || !X_clipped) {
        fprintf(stderr, "Error: Invalid pointers in limit_bounds\n");
        return;
    }
    for (int j = 0; j < size; j++) {
        X_clipped[j] = (X[j] < opt->bounds[2 * j]) ? opt->bounds[2 * j] : (X[j] > opt->bounds[2 * j + 1]) ? opt->bounds[2 * j + 1] : X[j];
        if (X_clipped[j] == opt->bounds[2 * j]) {
            double range_j = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
            X_clipped[j] += rand_double(0.0, 0.1 * range_j);
        } else if (X_clipped[j] == opt->bounds[2 * j + 1]) {
            double range_j = opt->bounds[2 * j + 1] - opt->bounds[2 * j];
            X_clipped[j] -= rand_double(0.0, 0.1 * range_j);
        }
    }
}

// Main Optimization Function
void COA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->best_solution.position || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer or memory allocation\n");
        return;
    }
    initialize_coyotes(opt, objective_function);
    int year = 1;
    int nfeval = 0;  // Local counter for function evaluations
    int max_packs = opt->population_size / N_COYOTES_PER_PACK;

    while (nfeval < MAX_NFEVAL) {
        for (int p = 0; p < N_PACKS && p < max_packs; p++) {
            update_pack(opt, p, objective_function);
        }
        pack_exchange(opt);
        int ibest = 0;
        for (int i = 0; i < N_PACKS * N_COYOTES_PER_PACK && i < opt->population_size; i++) {
            if (opt->population[i].fitness < opt->population[ibest].fitness) {
                ibest = i;
            }
        }
        if (opt->population[ibest].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[ibest].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[ibest].position[j];
            }
        }
        printf("Year %d: Best Value = %f\n", year++, opt->best_solution.fitness);
        nfeval += N_PACKS * N_COYOTES_PER_PACK;  // Approximate nfeval increment
    }
}
