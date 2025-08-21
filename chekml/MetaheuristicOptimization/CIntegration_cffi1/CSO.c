#include "CSO.h"
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <time.h>

// Enable debug logging via compile-time flag
#define CSO_DEBUG 0
#if CSO_DEBUG
#define DEBUG_LOG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// Fast Xorshift RNG
typedef struct {
    uint32_t state;
} Xorshift;

static inline uint32_t xorshift_next(Xorshift *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static inline double xorshift_double(Xorshift *rng, double min, double max) {
    return min + (max - min) * (xorshift_next(rng) / (double)UINT32_MAX);
}

// CSO Buffers
struct CSO_Buffers {
    int *sort_indices;
    int *mate;
    int *mother_lib;
    int *temp_perm;
    double *positions; // Contiguous population positions
    double *fitnesses; // Contiguous fitness values
    Xorshift rng;
};

// Heap sort for indices based on fitness
static void heapify(int *indices, double *fitnesses, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && fitnesses[indices[left]] < fitnesses[indices[largest]]) {
        largest = left;
    }
    if (right < n && fitnesses[indices[right]] < fitnesses[indices[largest]]) {
        largest = right;
    }
    if (largest != i) {
        int temp = indices[i];
        indices[i] = indices[largest];
        indices[largest] = temp;
        heapify(indices, fitnesses, n, largest);
    }
}

static void heap_sort(int *indices, double *fitnesses, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(indices, fitnesses, n, i);
    }
    for (int i = n - 1; i > 0; i--) {
        int temp = indices[0];
        indices[0] = indices[i];
        indices[i] = temp;
        heapify(indices, fitnesses, i, 0);
    }
}

// Random integer excluding tabu
static inline int randi_tabu(Xorshift *rng, int min_val, int max_val, int tabu) {
    int temp;
    do {
        temp = min_val + (int)((max_val - min_val + 1) * xorshift_double(rng, 0.0, 1.0));
    } while (temp == tabu);
    return temp;
}

// Optimized permutation
static void randperm_f(Xorshift *rng, int range_size, int dim, int *result, int *temp) {
    for (int i = 0; i < range_size; i++) {
        temp[i] = i + 1;
    }
    for (int i = range_size - 1; i > 0; i--) {
        int j = (int)(xorshift_double(rng, 0.0, 1.0) * (i + 1));
        int swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
    }
    for (int i = 0; i < dim; i++) {
        result[i] = (i < range_size) ? temp[i] : (1 + (int)(xorshift_double(rng, 0.0, 1.0) * range_size));
    }
}

// Initialize population
void initialize_population_cso(Optimizer *opt, double (*objective_function)(double *), CSO_Buffers *buffers) {
    // Validate bounds
    for (int j = 0; j < opt->dim; j++) {
        if (opt->bounds[2*j] > opt->bounds[2*j+1]) {
            DEBUG_LOG("Error: Invalid bounds at dimension %d: [%f, %f]\n",
                      j, opt->bounds[2*j], opt->bounds[2*j+1]);
            exit(1);
        }
    }

    double *positions = buffers->positions;
    double *fitnesses = buffers->fitnesses;
    for (int i = 0; i < opt->population_size; i++) {
        double *pos = positions + i * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] = opt->bounds[2*j] + xorshift_double(&buffers->rng, 0.0, 1.0) * (opt->bounds[2*j+1] - opt->bounds[2*j]);
        }
        fitnesses[i] = objective_function(pos);
        if (isnan(fitnesses[i]) || isinf(fitnesses[i])) {
            DEBUG_LOG("Warning: Invalid initial fitness at population[%d]: %f, position=[", i, fitnesses[i]);
            for (int j = 0; j < opt->dim; j++) {
                DEBUG_LOG("%f ", pos[j]);
            }
            DEBUG_LOG("]\n");
            fitnesses[i] = DBL_MAX;
        }
        opt->population[i].position = pos;
        opt->population[i].fitness = fitnesses[i];
    }
    enforce_bound_constraints(opt);

    int best_idx = 0;
    for (int i = 1; i < opt->population_size; i++) {
        if (fitnesses[i] < fitnesses[best_idx]) {
            best_idx = i;
        }
    }
    opt->best_solution.fitness = fitnesses[best_idx];
    memcpy(opt->best_solution.position, positions + best_idx * opt->dim, opt->dim * sizeof(double));
}

// Update roosters
static void update_roosters(Optimizer *opt, CSO_Buffers *buffers, int rooster_num) {
    int *sort_indices = buffers->sort_indices;
    double *positions = buffers->positions;
    double *fitnesses = buffers->fitnesses;

    for (int i = 0; i < rooster_num; i++) {
        int curr_idx = sort_indices[i];
        int another_rooster = randi_tabu(&buffers->rng, 1, rooster_num, i + 1);
        int another_idx = sort_indices[another_rooster - 1];

        double sigma = 1.0;
        double curr_fitness = fitnesses[curr_idx];
        double other_fitness = fitnesses[another_idx];
        if (curr_fitness > other_fitness && isfinite(curr_fitness) && isfinite(other_fitness)) {
            double diff = other_fitness - curr_fitness;
            double denom = fabs(curr_fitness) + 2.2e-16;
            if (fabs(diff / denom) < 100.0) {
                sigma = exp(diff / denom);
            }
        }

        double *pos = positions + curr_idx * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            pos[j] *= 1.0 + sigma * xorshift_double(&buffers->rng, -3.0, 3.0);
            if (isnan(pos[j]) || isinf(pos[j])) {
                DEBUG_LOG("Warning: Invalid position at rooster[%d][%d]: %f\n", curr_idx, j, pos[j]);
                pos[j] = opt->bounds[2*j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update hens
static void update_hens(Optimizer *opt, CSO_Buffers *buffers, int rooster_num, int hen_num) {
    int *sort_indices = buffers->sort_indices;
    int *mate = buffers->mate;
    double *positions = buffers->positions;
    double *fitnesses = buffers->fitnesses;

    for (int i = rooster_num; i < rooster_num + hen_num; i++) {
        int curr_idx = sort_indices[i];
        int mate_idx = sort_indices[mate[i - rooster_num] - 1];
        int other = randi_tabu(&buffers->rng, 1, i, mate[i - rooster_num]);
        int other_idx = sort_indices[other - 1];

        double curr_fitness = fitnesses[curr_idx];
        double mate_fitness = fitnesses[mate_idx];
        double other_fitness = fitnesses[other_idx];

        double c1 = 1.0, c2 = 1.0;
        if (isfinite(curr_fitness) && isfinite(mate_fitness) && isfinite(other_fitness)) {
            double diff1 = curr_fitness - mate_fitness;
            double denom1 = fabs(curr_fitness) + 2.2e-16;
            if (fabs(diff1 / denom1) < 100.0) {
                c1 = exp(diff1 / denom1);
            }
            double diff2 = other_fitness - curr_fitness;
            if (fabs(diff2) < 100.0) {
                c2 = exp(diff2);
            }
        }

        double *curr_pos = positions + curr_idx * opt->dim;
        double *mate_pos = positions + mate_idx * opt->dim;
        double *other_pos = positions + other_idx * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            curr_pos[j] += c1 * xorshift_double(&buffers->rng, 0.0, 1.0) * (mate_pos[j] - curr_pos[j]) +
                           c2 * xorshift_double(&buffers->rng, 0.0, 1.0) * (other_pos[j] - curr_pos[j]);
            if (isnan(curr_pos[j]) || isinf(curr_pos[j])) {
                DEBUG_LOG("Warning: Invalid position at hen[%d][%d]: %f\n", curr_idx, j, curr_pos[j]);
                curr_pos[j] = opt->bounds[2*j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update chicks
static void update_chicks(Optimizer *opt, CSO_Buffers *buffers, int rooster_num, int hen_num, int mother_num) {
    int *sort_indices = buffers->sort_indices;
    int *mother_lib = buffers->mother_lib;
    double *positions = buffers->positions;

    int chick_num = opt->population_size - rooster_num - hen_num;
    for (int i = rooster_num + hen_num; i < opt->population_size; i++) {
        int curr_idx = sort_indices[i];
        int mother_idx = sort_indices[mother_lib[(int)(xorshift_double(&buffers->rng, 0.0, 1.0) * mother_num)] - 1];
        double fl = 0.5 + 0.4 * xorshift_double(&buffers->rng, 0.0, 1.0);

        double *curr_pos = positions + curr_idx * opt->dim;
        double *mother_pos = positions + mother_idx * opt->dim;
        for (int j = 0; j < opt->dim; j++) {
            curr_pos[j] += fl * (mother_pos[j] - curr_pos[j]);
            if (isnan(curr_pos[j]) || isinf(curr_pos[j])) {
                DEBUG_LOG("Warning: Invalid position at chick[%d][%d]: %f\n", curr_idx, j, curr_pos[j]);
                curr_pos[j] = opt->bounds[2*j];
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Update best solutions
static void update_best_solutions(Optimizer *opt, double (*objective_function)(double *), CSO_Buffers *buffers) {
    double *positions = buffers->positions;
    double *fitnesses = buffers->fitnesses;
    double best_fitness = opt->best_solution.fitness;

    for (int i = 0; i < opt->population_size; i++) {
        double *pos = positions + i * opt->dim;
        fitnesses[i] = objective_function(pos);
        if (isnan(fitnesses[i]) || isinf(fitnesses[i])) {
            DEBUG_LOG("Warning: Invalid fitness at population[%d]: %f, position=[", i, fitnesses[i]);
            for (int j = 0; j < opt->dim; j++) {
                DEBUG_LOG("%f ", pos[j]);
            }
            DEBUG_LOG("]\n");
            fitnesses[i] = DBL_MAX;
        }
        opt->population[i].fitness = fitnesses[i];
        if (fitnesses[i] < best_fitness) {
            best_fitness = fitnesses[i];
            opt->best_solution.fitness = fitnesses[i];
            memcpy(opt->best_solution.position, pos, opt->dim * sizeof(double));
        }
    }
}

// Main Optimization Function
void CSO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize RNG
    CSO_Buffers buffers = {0};
    buffers.rng.state = (uint32_t)time(NULL) | 1; // Ensure non-zero seed

    // Pre-compute constants
    int rooster_num = (int)(opt->population_size * ROOSTER_RATIO);
    int hen_num = (int)(opt->population_size * HEN_RATIO);
    int mother_num = (int)(hen_num * MOTHER_RATIO);

    // Allocate buffers
    buffers.sort_indices = (int *)malloc(opt->population_size * sizeof(int));
    buffers.mate = (int *)malloc(hen_num * sizeof(int));
    buffers.mother_lib = (int *)malloc(mother_num * sizeof(int));
    buffers.temp_perm = (int *)malloc(opt->population_size * sizeof(int));
    buffers.positions = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    buffers.fitnesses = (double *)malloc(opt->population_size * sizeof(double));
    if (!buffers.sort_indices || !buffers.mate || !buffers.mother_lib ||
        !buffers.temp_perm || !buffers.positions || !buffers.fitnesses) {
        DEBUG_LOG("Error: Memory allocation failed\n");
        free(buffers.sort_indices);
        free(buffers.mate);
        free(buffers.mother_lib);
        free(buffers.temp_perm);
        free(buffers.positions);
        free(buffers.fitnesses);
        exit(1);
    }

    initialize_population_cso(opt, objective_function, &buffers);

    for (int t = 0; t < opt->max_iter; t++) {
        if (t % UPDATE_FREQ == 0 || t == 0) {
            for (int i = 0; i < opt->population_size; i++) {
                buffers.sort_indices[i] = i;
            }
            heap_sort(buffers.sort_indices, buffers.fitnesses, opt->population_size);

            randperm_f(&buffers.rng, rooster_num, hen_num, buffers.mate, buffers.temp_perm);
            randperm_f(&buffers.rng, hen_num, mother_num, buffers.mother_lib, buffers.temp_perm);
            for (int i = 0; i < mother_num; i++) {
                buffers.mother_lib[i] += rooster_num;
            }
        }

        update_roosters(opt, &buffers, rooster_num);
        update_hens(opt, &buffers, rooster_num, hen_num);
        update_chicks(opt, &buffers, rooster_num, hen_num, mother_num);
        update_best_solutions(opt, objective_function, &buffers);

        printf("Iteration %d: Best Value = %f\n", t + 1, opt->best_solution.fitness);
    }

    free(buffers.sort_indices);
    free(buffers.mate);
    free(buffers.mother_lib);
    free(buffers.temp_perm);
    free(buffers.positions);
    free(buffers.fitnesses);
}
