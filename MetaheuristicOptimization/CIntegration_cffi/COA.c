#include "COA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define RAND_MAX_D (RAND_MAX + 1.0)

// Inline random double generator
static inline double rand_double(double min, double max) {
    return min + (max - min) * (rand() / RAND_MAX_D);
}

// Helper function for qsort
int compare_doubles_coa(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    return (arg1 < arg2) ? -1 : (arg1 > arg2) ? 1 : 0;
}

// Initialize coyote population
void initialize_coyotes(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !opt->population || !opt->best_solution.position || !opt->bounds) {
        fprintf(stderr, "Error: Invalid Optimizer or memory allocation\n");
        return;
    }
    srand(time(NULL));
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
        double *pos = opt->population[c].position;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[c].fitness = objective_function(pos);
        nfeval++;
    }

    int ibest = 0;
    for (int i = 1; i < total_coyotes && i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[ibest].fitness) {
            ibest = i;
        }
    }
    opt->best_solution.fitness = opt->population[ibest].fitness;
    double *best_pos = opt->best_solution.position;
    double *ibest_pos = opt->population[ibest].position;
    for (int j = 0; j < opt->dim; j++) {
        best_pos[j] = ibest_pos[j];
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
    double pack_ages[N_COYOTES_PER_PACK];
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
        pack_ages[i] = i + 1;
    }

    // Sort pack by fitness (bubble sort optimized for small N)
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
    Solution *pop = opt->population;
    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        pop[start_idx + i].fitness = pack_costs[i];
        for (int j = 0; j < opt->dim; j++) {
            pop[start_idx + i].position[j] = pop[start_idx + ind[i]].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        c_alpha[j] = pop[start_idx].position[j];
    }

    // Calculate tendency (median)
    for (int j = 0; j < opt->dim; j++) {
        double temp[N_COYOTES_PER_PACK];
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            temp[i] = pop[start_idx + i].position[j];
        }
        qsort(temp, N_COYOTES_PER_PACK, sizeof(double), compare_doubles_coa);
        tendency[j] = (N_COYOTES_PER_PACK % 2 == 0) ? (temp[2] + temp[3]) / 2.0 : temp[2]; // Unroll for N=5
    }

    // Diversity and scale factor (optimize by reusing mean)
    double diversity = 0.0, mean, std;
    for (int j = 0; j < opt->dim; j++) {
        mean = 0.0;
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            mean += pop[start_idx + i].position[j];
        }
        mean /= N_COYOTES_PER_PACK;
        std = 0.0;
        for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
            double diff = pop[start_idx + i].position[j] - mean;
            std += diff * diff; // Avoid pow
        }
        std = sqrt(std / N_COYOTES_PER_PACK);
        diversity += std;
    }
    diversity /= opt->dim;
    double scale_factor = (diversity < SCALE_FACTOR_DIVERSITY_THRESHOLD) ? SCALE_FACTOR_HIGH : SCALE_FACTOR_LOW;

    // Update coyotes with pre-computed randoms
    double rands[N_COYOTES_PER_PACK * 2];
    for (int i = 0; i < N_COYOTES_PER_PACK * 2; i++) rands[i] = rand_double(0.0, 1.0);
    for (int c = 0; c < N_COYOTES_PER_PACK; c++) {
        int rc1 = c, rc2 = c;
        while (rc1 == c) rc1 = rand() % N_COYOTES_PER_PACK;
        while (rc2 == c || rc2 == rc1) rc2 = rand() % N_COYOTES_PER_PACK;

        for (int j = 0; j < opt->dim; j++) {
            new_coyotes[c][j] = pop[start_idx + c].position[j] +
                                scale_factor * rands[2 * c] * (c_alpha[j] - pop[start_idx + rc1].position[j]) +
                                scale_factor * rands[2 * c + 1] * (tendency[j] - pop[start_idx + rc2].position[j]);
        }
        limit_bounds(opt, new_coyotes[c], pop[start_idx + c].position, opt->dim);

        double new_cost = objective_function(pop[start_idx + c].position);
        nfeval++;
        if (new_cost < pack_costs[c]) {
            pack_costs[c] = new_cost;
            for (int j = 0; j < opt->dim; j++) {
                pop[start_idx + c].position[j] = new_coyotes[c][j];
            }
        }
    }

    // Breeding (simplified)
    int parents[2];
    for (int i = 0; i < 2; i++) parents[i] = rand() % N_COYOTES_PER_PACK;
    double pup[opt->dim];
    double r = rand_double(0.0, 1.0);
    for (int j = 0; j < opt->dim; j++) {
        if (r < PS / 2) pup[j] = pop[start_idx + parents[0]].position[j];
        else if (r < 1.0 - PS / 2) pup[j] = pop[start_idx + parents[1]].position[j];
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
            pop[start_idx + worst_idx].position[j] = pup[j];
        }
    }

    for (int i = 0; i < N_COYOTES_PER_PACK; i++) {
        pop[start_idx + i].fitness = pack_costs[i];
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
        double *pos1 = opt->population[idx1].position;
        double *pos2 = opt->population[idx2].position;
        for (int j = 0; j < opt->dim; j++) {
            double temp = pos1[j];
            pos1[j] = pos2[j];
            pos2[j] = temp;
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
    int nfeval = 0;
    int max_packs = opt->population_size / N_COYOTES_PER_PACK;

    while (nfeval < MAX_NFEVAL) {
        #pragma omp parallel for if(max_packs > 1) schedule(static)
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
            double *best_pos = opt->best_solution.position;
            double *ibest_pos = opt->population[ibest].position;
            for (int j = 0; j < opt->dim; j++) {
                best_pos[j] = ibest_pos[j];
            }
        }
        printf("Year %d: Best Value = %f\n", year++, opt->best_solution.fitness);
        nfeval += N_PACKS * N_COYOTES_PER_PACK;
    }
}
