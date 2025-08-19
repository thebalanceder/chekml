#include "COA.h"
#include "generaloptimizer.h"
#include <emmintrin.h> // SSE2 intrinsics
#include <stdint.h>
#include <stdio.h>
#include <time.h> // Added for time()

// Constants
#define N_COYOTES_PER_PACK 5
#define N_PACKS 10
#define MAX_NFEVAL 20000
#define P_LEAVE 0.01
#define PS 1.0
#define SCALE_FACTOR_DIVERSITY_THRESHOLD 0.1
#define SCALE_FACTOR_HIGH 1.5
#define SCALE_FACTOR_LOW 1.0
#define DIM 2 // Hardcoded for SSE2 vectorization (adjust if different)

static uint32_t xorshift_state = 123456789;

// Fast inline Xorshift PRNG
static inline double xorshift_double(double min, double max) {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    return min + (max - min) * (xorshift_state / 4294967296.0);
}

// Helper function for qsort
static int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    return (arg1 < arg2) ? -1 : (arg1 > arg2) ? 1 : 0;
}

// Initialize coyote population (unrolled)
void initialize_coyotes(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));
    int total_coyotes = N_PACKS * N_COYOTES_PER_PACK;
    int nfeval = 0;
    __m128d bounds_low = _mm_loadu_pd(opt->bounds);
    __m128d bounds_high = _mm_loadu_pd(opt->bounds + 2);
    __m128d range = _mm_sub_pd(bounds_high, bounds_low);

    for (int c = 0; c < total_coyotes && c < opt->population_size; c++) {
        double *pos = opt->population[c].position;
        __m128d rand_vec = _mm_set_pd(xorshift_double(0.0, 1.0), xorshift_double(0.0, 1.0));
        __m128d scaled_rand = _mm_mul_pd(rand_vec, range);
        __m128d result = _mm_add_pd(bounds_low, scaled_rand);
        _mm_storeu_pd(pos, result);
        pos[2] = xorshift_double(opt->bounds[4], opt->bounds[5]); // Handle dim > 2 manually if needed
        opt->population[c].fitness = objective_function(pos);
        nfeval++;
    }

    int ibest = 0;
    for (int i = 1; i < total_coyotes && i < opt->population_size; i++) {
        if (opt->population[i].fitness < opt->population[ibest].fitness) ibest = i;
    }
    opt->best_solution.fitness = opt->population[ibest].fitness;
    double *best_pos = opt->best_solution.position;
    double *ibest_pos = opt->population[ibest].position;
    best_pos[0] = ibest_pos[0];
    best_pos[1] = ibest_pos[1];
}

// Update a single pack (unrolled)
void update_pack(Optimizer *opt, int pack_idx, double (*objective_function)(double *)) {
    int start_idx = pack_idx * N_COYOTES_PER_PACK;
    Solution *pop = opt->population;
    double pack_costs[5] = {pop[start_idx].fitness, pop[start_idx + 1].fitness, pop[start_idx + 2].fitness,
                            pop[start_idx + 3].fitness, pop[start_idx + 4].fitness};
    int ind[5] = {0, 1, 2, 3, 4};

    // Bubble sort unrolled
    if (pack_costs[0] > pack_costs[1]) { int t = ind[0]; ind[0] = ind[1]; ind[1] = t; double tc = pack_costs[0]; pack_costs[0] = pack_costs[1]; pack_costs[1] = tc; }
    if (pack_costs[1] > pack_costs[2]) { int t = ind[1]; ind[1] = ind[2]; ind[2] = t; double tc = pack_costs[1]; pack_costs[1] = pack_costs[2]; pack_costs[2] = tc; }
    if (pack_costs[2] > pack_costs[3]) { int t = ind[2]; ind[2] = ind[3]; ind[3] = t; double tc = pack_costs[2]; pack_costs[2] = pack_costs[3]; pack_costs[3] = tc; }
    if (pack_costs[3] > pack_costs[4]) { int t = ind[3]; ind[3] = ind[4]; ind[4] = t; double tc = pack_costs[3]; pack_costs[3] = pack_costs[4]; pack_costs[4] = tc; }
    if (pack_costs[0] > pack_costs[1]) { int t = ind[0]; ind[0] = ind[1]; ind[1] = t; double tc = pack_costs[0]; pack_costs[0] = pack_costs[1]; pack_costs[1] = tc; }
    if (pack_costs[1] > pack_costs[2]) { int t = ind[1]; ind[1] = ind[2]; ind[2] = t; double tc = pack_costs[1]; pack_costs[1] = pack_costs[2]; pack_costs[2] = tc; }
    if (pack_costs[2] > pack_costs[3]) { int t = ind[2]; ind[2] = ind[3]; ind[3] = t; double tc = pack_costs[2]; pack_costs[2] = pack_costs[3]; pack_costs[3] = tc; }

    // Reorder
    double temp_costs[5], temp_ages[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) { temp_costs[i] = pack_costs[ind[i]]; temp_ages[i] = temp_ages[ind[i]]; }
    for (int i = 0; i < 5; i++) { pack_costs[i] = temp_costs[i]; }
    for (int i = 0; i < 5; i++) {
        pop[start_idx + i].fitness = pack_costs[i];
        pop[start_idx + i].position[0] = pop[start_idx + ind[i]].position[0];
        pop[start_idx + i].position[1] = pop[start_idx + ind[i]].position[1];
    }
    double c_alpha[2] = {pop[start_idx].position[0], pop[start_idx].position[1]};

    // Tendency (median for 5 elements)
    double temp[5];
    for (int i = 0; i < 5; i++) temp[i] = pop[start_idx + i].position[0];
    qsort(temp, 5, sizeof(double), compare_doubles);
    double tendency0 = (temp[2] + temp[3]) / 2.0;
    for (int i = 0; i < 5; i++) temp[i] = pop[start_idx + i].position[1];
    qsort(temp, 5, sizeof(double), compare_doubles);
    double tendency1 = (temp[2] + temp[3]) / 2.0;

    // Diversity
    double mean0 = (pop[start_idx].position[0] + pop[start_idx + 1].position[0] + pop[start_idx + 2].position[0] +
                    pop[start_idx + 3].position[0] + pop[start_idx + 4].position[0]) / 5.0;
    double mean1 = (pop[start_idx].position[1] + pop[start_idx + 1].position[1] + pop[start_idx + 2].position[1] +
                    pop[start_idx + 3].position[1] + pop[start_idx + 4].position[1]) / 5.0;
    double std0 = 0.0, std1 = 0.0;
    std0 += (pop[start_idx].position[0] - mean0) * (pop[start_idx].position[0] - mean0);
    std0 += (pop[start_idx + 1].position[0] - mean0) * (pop[start_idx + 1].position[0] - mean0);
    std0 += (pop[start_idx + 2].position[0] - mean0) * (pop[start_idx + 2].position[0] - mean0);
    std0 += (pop[start_idx + 3].position[0] - mean0) * (pop[start_idx + 3].position[0] - mean0);
    std0 += (pop[start_idx + 4].position[0] - mean0) * (pop[start_idx + 4].position[0] - mean0);
    std1 += (pop[start_idx].position[1] - mean1) * (pop[start_idx].position[1] - mean1);
    std1 += (pop[start_idx + 1].position[1] - mean1) * (pop[start_idx + 1].position[1] - mean1);
    std1 += (pop[start_idx + 2].position[1] - mean1) * (pop[start_idx + 2].position[1] - mean1);
    std1 += (pop[start_idx + 3].position[1] - mean1) * (pop[start_idx + 3].position[1] - mean1);
    std1 += (pop[start_idx + 4].position[1] - mean1) * (pop[start_idx + 4].position[1] - mean1);
    double diversity = sqrt(std0 / 5.0) + sqrt(std1 / 5.0);
    double scale_factor = (diversity < SCALE_FACTOR_DIVERSITY_THRESHOLD) ? SCALE_FACTOR_HIGH : SCALE_FACTOR_LOW;

    // Update coyotes
    double rands[10];
    for (int i = 0; i < 10; i++) rands[i] = xorshift_double(0.0, 1.0);
    double new_coyotes[5][2];
    for (int c = 0; c < 5; c++) {
        int rc1 = c, rc2 = c;
        while (rc1 == c) rc1 = rand() % 5;
        while (rc2 == c || rc2 == rc1) rc2 = rand() % 5;
        new_coyotes[c][0] = pop[start_idx + c].position[0] +
                            scale_factor * rands[2 * c] * (c_alpha[0] - pop[start_idx + rc1].position[0]) +
                            scale_factor * rands[2 * c + 1] * (tendency0 - pop[start_idx + rc2].position[0]);
        new_coyotes[c][1] = pop[start_idx + c].position[1] +
                            scale_factor * rands[2 * c] * (c_alpha[1] - pop[start_idx + rc1].position[1]) +
                            scale_factor * rands[2 * c + 1] * (tendency1 - pop[start_idx + rc2].position[1]);
        limit_bounds(opt, new_coyotes[c], pop[start_idx + c].position, DIM);
        double new_cost = objective_function(pop[start_idx + c].position);
        if (new_cost < pack_costs[c]) {
            pack_costs[c] = new_cost;
            pop[start_idx + c].position[0] = new_coyotes[c][0];
            pop[start_idx + c].position[1] = new_coyotes[c][1];
        }
    }

    // Breeding
    int parents[2] = {rand() % 5, rand() % 5};
    double pup[2], r = xorshift_double(0.0, 1.0);
    pup[0] = (r < PS / 2) ? pop[start_idx + parents[0]].position[0] : (r < 1.0 - PS / 2) ? pop[start_idx + parents[1]].position[0] : xorshift_double(opt->bounds[0], opt->bounds[1]);
    pup[1] = (r < PS / 2) ? pop[start_idx + parents[0]].position[1] : (r < 1.0 - PS / 2) ? pop[start_idx + parents[1]].position[1] : xorshift_double(opt->bounds[2], opt->bounds[3]);
    limit_bounds(opt, pup, pup, DIM);
    double pup_cost = objective_function(pup);
    int worst_idx = 0;
    if (pack_costs[1] > pack_costs[0]) worst_idx = 1;
    if (pack_costs[2] > pack_costs[worst_idx]) worst_idx = 2;
    if (pack_costs[3] > pack_costs[worst_idx]) worst_idx = 3;
    if (pack_costs[4] > pack_costs[worst_idx]) worst_idx = 4;
    if (pup_cost < pack_costs[worst_idx]) {
        pack_costs[worst_idx] = pup_cost;
        pop[start_idx + worst_idx].position[0] = pup[0];
        pop[start_idx + worst_idx].position[1] = pup[1];
    }

    for (int i = 0; i < 5; i++) pop[start_idx + i].fitness = pack_costs[i];
}

// Exchange coyotes between packs
void pack_exchange(Optimizer *opt) {
    if (N_PACKS > 1 && xorshift_double(0.0, 1.0) < P_LEAVE) {
        int rp[2] = {rand() % N_PACKS, rand() % N_PACKS};
        while (rp[1] == rp[0]) rp[1] = rand() % N_PACKS;
        int rc[2] = {rand() % N_COYOTES_PER_PACK, rand() % N_COYOTES_PER_PACK};
        int idx1 = rp[0] * N_COYOTES_PER_PACK + rc[0];
        int idx2 = rp[1] * N_COYOTES_PER_PACK + rc[1];
        double *pos1 = opt->population[idx1].position;
        double *pos2 = opt->population[idx2].position;
        double temp0 = pos1[0]; pos1[0] = pos2[0]; pos2[0] = temp0;
        double temp1 = pos1[1]; pos1[1] = pos2[1]; pos2[1] = temp1;
    }
}

// Limit bounds (unrolled for DIM=2)
void limit_bounds(Optimizer *opt, double *X, double *X_clipped, int size) {
    X_clipped[0] = (X[0] < opt->bounds[0]) ? opt->bounds[0] : (X[0] > opt->bounds[1]) ? opt->bounds[1] : X[0];
    if (X_clipped[0] == opt->bounds[0]) X_clipped[0] += xorshift_double(0.0, 0.1 * (opt->bounds[1] - opt->bounds[0]));
    else if (X_clipped[0] == opt->bounds[1]) X_clipped[0] -= xorshift_double(0.0, 0.1 * (opt->bounds[1] - opt->bounds[0]));
    X_clipped[1] = (X[1] < opt->bounds[2]) ? opt->bounds[2] : (X[1] > opt->bounds[3]) ? opt->bounds[3] : X[1];
    if (X_clipped[1] == opt->bounds[2]) X_clipped[1] += xorshift_double(0.0, 0.1 * (opt->bounds[3] - opt->bounds[2]));
    else if (X_clipped[1] == opt->bounds[3]) X_clipped[1] -= xorshift_double(0.0, 0.1 * (opt->bounds[3] - opt->bounds[2]));
}

// Main Optimization Function
void COA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_coyotes(opt, objective_function);
    int year = 1;
    int nfeval = 0;

    while (nfeval < MAX_NFEVAL) {
        for (int p = 0; p < N_PACKS; p++) update_pack(opt, p, objective_function);
        pack_exchange(opt);
        int ibest = 0;
        for (int i = 0; i < N_PACKS * N_COYOTES_PER_PACK; i++) {
            if (opt->population[i].fitness < opt->population[ibest].fitness) ibest = i;
        }
        if (opt->population[ibest].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[ibest].fitness;
            opt->best_solution.position[0] = opt->population[ibest].position[0];
            opt->best_solution.position[1] = opt->population[ibest].position[1];
        }
        printf("Year %d: Best Value = %f\n", year++, opt->best_solution.fitness);
        nfeval += N_PACKS * N_COYOTES_PER_PACK;
    }
}
