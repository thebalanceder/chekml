#include "SOA.h"
#include <emmintrin.h> // SSE2 intrinsics
#include <stdint.h>
#include <string.h>

// Fallback definitions for W_MAX, W_MIN if not in SOA.h
#ifndef SOA_W_MAX
#define SOA_W_MAX 0.9
#endif
#ifndef SOA_W_MIN
#define SOA_W_MIN 0.2
#endif

// Compile-time debug flag
#ifdef SOA_DEBUG
#include <stdio.h>
#define DEBUG_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_LOG(...)
#endif

#define SOA_MAGIC 0x50A50A50A50A50A5

// Fast XOR-shift RNG state
typedef struct {
    uint64_t state;
} XORShiftRNG;

// Initialize RNG
static inline void xorshift_init(XORShiftRNG *rng, uint64_t seed) {
    rng->state = seed ? seed : 88172645463325252ULL;
}

// Generate random double in [0, 1)
static inline double xorshift_random(XORShiftRNG *rng) {
    rng->state ^= rng->state >> 12;
    rng->state ^= rng->state << 25;
    rng->state ^= rng->state >> 27;
    return ((rng->state * 2685821657736338717ULL) >> 32) / 4294967296.0;
}

// Temporary arrays for optimization
typedef struct {
    double *mu_s;
    double *x_pdirect;
    double *x_ldirect1;
    double *x_ldirect2;
    double *x_tdirect;
    double *num_pone;
    double *num_none;
    double *x_direct;
    double *r_temp;
    double *en_temp;
    double *temp_population;
    int *sorted_indices;
    int *indices_changed;
    double *random_buffer;
    int random_count;
    int random_capacity;
    XORShiftRNG rng;
} SOATemps;

// Initialize SOA
SOA* SOA_init(Optimizer *opt) {
    if (!opt || opt->dim <= 0 || opt->dim > 1000 || 
        opt->population_size <= 0 || opt->population_size > 10000 ||
        opt->max_iter <= 0 || opt->max_iter > 100000) {
        DEBUG_LOG("SOA_init: Invalid optimizer parameters\n");
        return NULL;
    }

    SOA *soa = (SOA*)calloc(1, sizeof(SOA));
    if (!soa) {
        DEBUG_LOG("SOA_init: Memory allocation failed for SOA\n");
        return NULL;
    }

    soa->magic = SOA_MAGIC;
    soa->opt = opt;
    soa->num_regions = NUM_REGIONS;
    soa->mu_max = SOA_MU_MAX;
    soa->mu_min = SOA_MU_MIN;
    soa->w_max = SOA_W_MAX;
    soa->w_min = SOA_W_MIN;

    // Allocate arrays
    size_t total_size = soa->num_regions * sizeof(int) * 3 + // start_reg, end_reg, size_reg
                        opt->dim * sizeof(double) * 2 +      // rmax, rmin
                        opt->population_size * opt->dim * sizeof(double) * 3 + // pbest_s, e_t_1, e_t_2
                        soa->num_regions * opt->dim * sizeof(double) +         // lbest_s
                        opt->population_size * sizeof(double) * 3 +            // pbest_fun, f_t_1, f_t_2
                        soa->num_regions * sizeof(double);                    // lbest_fun
    void *block = calloc(1, total_size);
    if (!block) {
        DEBUG_LOG("SOA_init: Memory allocation failed for block\n");
        free(soa);
        return NULL;
    }

    char *ptr = (char*)block;
    soa->start_reg = (int*)ptr; ptr += soa->num_regions * sizeof(int);
    soa->end_reg = (int*)ptr; ptr += soa->num_regions * sizeof(int);
    soa->size_reg = (int*)ptr; ptr += soa->num_regions * sizeof(int);
    soa->rmax = (double*)ptr; ptr += opt->dim * sizeof(double);
    soa->rmin = (double*)ptr; ptr += opt->dim * sizeof(double);
    soa->pbest_s = (double*)ptr; ptr += opt->population_size * opt->dim * sizeof(double);
    soa->e_t_1 = (double*)ptr; ptr += opt->population_size * opt->dim * sizeof(double);
    soa->e_t_2 = (double*)ptr; ptr += opt->population_size * opt->dim * sizeof(double);
    soa->lbest_s = (double*)ptr; ptr += soa->num_regions * opt->dim * sizeof(double);
    soa->pbest_fun = (double*)ptr; ptr += opt->population_size * sizeof(double);
    soa->f_t_1 = (double*)ptr; ptr += opt->population_size * sizeof(double);
    soa->f_t_2 = (double*)ptr; ptr += soa->num_regions * sizeof(double);
    soa->lbest_fun = (double*)ptr;

    // Initialize regions
    for (int r = 0; r < soa->num_regions; r++) {
        soa->start_reg[r] = (r * opt->population_size) / soa->num_regions;
        soa->end_reg[r] = ((r + 1) * opt->population_size) / soa->num_regions;
        soa->size_reg[r] = soa->end_reg[r] - soa->start_reg[r];
        DEBUG_LOG("SOA_init: Region %d: start=%d, end=%d, size=%d\n",
                  r, soa->start_reg[r], soa->end_reg[r], soa->size_reg[r]);
    }

    // Initialize step sizes
    for (int j = 0; j < opt->dim; j++) {
        if (!opt->bounds) {
            DEBUG_LOG("SOA_init: NULL bounds array\n");
            SOA_free(soa);
            return NULL;
        }
        soa->rmax[j] = 0.5 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        soa->rmin[j] = -soa->rmax[j];
    }

    // Initialize population and bests
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population || !opt->population[i].position) {
            DEBUG_LOG("SOA_init: NULL population or population[%d].position\n", i);
            SOA_free(soa);
            return NULL;
        }
        memcpy(soa->pbest_s + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        memcpy(soa->e_t_1 + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        memcpy(soa->e_t_2 + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        soa->pbest_fun[i] = INFINITY;
        soa->f_t_1[i] = INFINITY;
        soa->f_t_2[i] = INFINITY;
    }
    for (int r = 0; r < soa->num_regions; r++) {
        memset(soa->lbest_s + r * opt->dim, 0, opt->dim * sizeof(double));
        soa->lbest_fun[r] = INFINITY;
    }

    DEBUG_LOG("SOA_init: Initialized SOA at %p\n", soa);
    return soa;
}

// Free SOA
void SOA_free(SOA *soa) {
    if (!soa) {
        DEBUG_LOG("SOA_free: NULL SOA pointer\n");
        return;
    }
    DEBUG_LOG("SOA_free: Freeing SOA at %p\n", soa);
    if (soa->start_reg) free(soa->start_reg); // Single block
    free(soa);
}

// Initialize SOATemps
static SOATemps* init_temps(int dim, int population_size, int num_regions) {
    SOATemps *temps = (SOATemps*)calloc(1, sizeof(SOATemps));
    if (!temps) {
        DEBUG_LOG("init_temps: Memory allocation failed for SOATemps\n");
        return NULL;
    }

    size_t total_size = dim * sizeof(double) * 10 +                      // mu_s, x_pdirect, etc.
                        population_size * dim * sizeof(double) +         // temp_population
                        population_size * sizeof(int) +                  // sorted_indices
                        num_regions * sizeof(int) +                      // indices_changed
                        population_size * 10 * sizeof(double);           // random_buffer
    void *block = calloc(1, total_size);
    if (!block) {
        DEBUG_LOG("init_temps: Memory allocation failed for block\n");
        free(temps);
        return NULL;
    }

    char *ptr = (char*)block;
    temps->mu_s = (double*)ptr; ptr += dim * sizeof(double);
    temps->x_pdirect = (double*)ptr; ptr += dim * sizeof(double);
    temps->x_ldirect1 = (double*)ptr; ptr += dim * sizeof(double);
    temps->x_ldirect2 = (double*)ptr; ptr += dim * sizeof(double);
    temps->x_tdirect = (double*)ptr; ptr += dim * sizeof(double);
    temps->num_pone = (double*)ptr; ptr += dim * sizeof(double);
    temps->num_none = (double*)ptr; ptr += dim * sizeof(double);
    temps->x_direct = (double*)ptr; ptr += dim * sizeof(double);
    temps->r_temp = (double*)ptr; ptr += dim * sizeof(double);
    temps->en_temp = (double*)ptr; ptr += dim * sizeof(double);
    temps->temp_population = (double*)ptr; ptr += population_size * dim * sizeof(double);
    temps->sorted_indices = (int*)ptr; ptr += population_size * sizeof(int);
    temps->indices_changed = (int*)ptr; ptr += num_regions * sizeof(int);
    temps->random_buffer = (double*)ptr;
    temps->random_count = 0;
    temps->random_capacity = population_size * 10;
    xorshift_init(&temps->rng, 123456789ULL);

    // Initialize sorted indices and indices_changed
    for (int r = 0, idx = 0; r < num_regions; r++) {
        int start = (r * population_size) / num_regions;
        int end = ((r + 1) * population_size) / num_regions;
        int size = end - start;
        for (int i = 0; i < size; i++) {
            temps->sorted_indices[idx++] = start + i;
        }
        temps->indices_changed[r] = 1;
    }

    return temps;
}

// Free SOATemps
static void free_temps(SOATemps *temps) {
    if (temps) {
        if (temps->mu_s) free(temps->mu_s); // Single block
        free(temps);
    }
}

// Main optimization function
void SOA_optimize(void *soa_ptr, ObjectiveFunction objective_function) {
    if (!soa_ptr || !objective_function) {
        DEBUG_LOG("SOA_optimize: Invalid arguments\n");
        return;
    }

    SOA *soa = (SOA*)soa_ptr;
    Optimizer *opt = (Optimizer*)soa_ptr;
    if (soa->magic != SOA_MAGIC) {
        DEBUG_LOG("SOA_optimize: Invalid SOA magic number: expected=%lx, got=%lx\n",
                  SOA_MAGIC, soa->magic);
        if (!opt->population || !opt->bounds || !opt->best_solution.position ||
            opt->dim <= 0 || opt->dim > 1000 ||
            opt->population_size <= 0 || opt->population_size > 10000 ||
            opt->max_iter <= 0 || opt->max_iter > 100000) {
            DEBUG_LOG("SOA_optimize: Invalid Optimizer parameters\n");
            return;
        }
        soa = SOA_init(opt);
        if (!soa) {
            DEBUG_LOG("SOA_optimize: Failed to initialize SOA\n");
            return;
        }
        DEBUG_LOG("SOA_optimize: Created temporary SOA at %p\n", soa);
    } else if (!soa->opt) {
        DEBUG_LOG("SOA_optimize: NULL optimizer in soa\n");
        return;
    } else {
        opt = soa->opt;
    }

    DEBUG_LOG("SOA_optimize: Starting optimization with dim=%d, population_size=%d, max_iter=%d\n",
              opt->dim, opt->population_size, opt->max_iter);

    // Initialize temps
    SOATemps *temps = init_temps(opt->dim, opt->population_size, soa->num_regions);
    if (!temps) {
        if (soa->magic != SOA_MAGIC) SOA_free(soa);
        return;
    }

    int max_fes = opt->max_iter * opt->population_size;
    int max_gens = opt->max_iter;
    int fes = 0, gens = 0;
    double error_prev = INFINITY;

    // Inline evaluate_population
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        fes++;
        if (fitness < soa->pbest_fun[i]) {
            soa->pbest_fun[i] = fitness;
            memcpy(soa->pbest_s + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        }
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
        }
    }
    enforce_bound_constraints(opt);

    // Inline update_region_bests
    for (int r = 0; r < soa->num_regions; r++) {
        if (soa->size_reg[r] <= 0) continue;
        int best_idx = soa->start_reg[r];
        double best_fun = opt->population[best_idx].fitness;
        for (int i = soa->start_reg[r]; i < soa->end_reg[r]; i++) {
            if (opt->population[i].fitness < best_fun) {
                best_fun = opt->population[i].fitness;
                best_idx = i;
            }
        }
        if (best_fun < soa->lbest_fun[r]) {
            soa->lbest_fun[r] = best_fun;
            memcpy(soa->lbest_s + r * opt->dim, opt->population[best_idx].position, opt->dim * sizeof(double));
        }
    }

    // Main optimization loop
    while (fes < max_fes) {
        gens++;
        double weight = soa->w_max - gens * (soa->w_max - soa->w_min) / max_gens;
        double mu = soa->mu_max - gens * (soa->mu_max - soa->mu_min) / max_gens;

        // Copy population
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(temps->temp_population + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        }

        // Process regions
        for (int r = 0, idx_offset = 0; r < soa->num_regions; r++) {
            if (soa->size_reg[r] <= 0) {
                idx_offset += soa->size_reg[r];
                continue;
            }

            // Sort indices if changed
            if (temps->indices_changed[r]) {
                int *indices = temps->sorted_indices + idx_offset;
                for (int i = 0; i < soa->size_reg[r] - 1; i++) {
                    for (int j = 0; j < soa->size_reg[r] - i - 1; j++) {
                        int idx1 = indices[j], idx2 = indices[j + 1];
                        if (opt->population[idx1].fitness < opt->population[idx2].fitness) {
                            int temp = indices[j];
                            indices[j] = indices[j + 1];
                            indices[j + 1] = temp;
                        }
                    }
                }
                temps->indices_changed[r] = 0;
            }

            // Compute exploration term (SIMD for dim=2)
            if (soa->size_reg[r] < 2) {
                idx_offset += soa->size_reg[r];
                continue;
            }
            int *indices = temps->sorted_indices + idx_offset;
            int rand_en = 1 + (int)(xorshift_random(&temps->rng) * (soa->size_reg[r] - 2));
            if (rand_en >= soa->size_reg[r]) rand_en = soa->size_reg[r] - 1;
            if (opt->dim == 2) {
                __m128d pos_worst = _mm_loadu_pd(opt->population[indices[soa->size_reg[r] - 1]].position);
                __m128d pos_rand = _mm_loadu_pd(opt->population[indices[rand_en]].position);
                __m128d diff = _mm_sub_pd(pos_worst, pos_rand);
                __m128d abs_diff = _mm_andnot_pd(_mm_set1_pd(-0.0), diff); // fabs
                __m128d w = _mm_set1_pd(weight);
                __m128d en = _mm_mul_pd(abs_diff, w);
                _mm_storeu_pd(temps->en_temp, en);
            } else {
                for (int j = 0; j < opt->dim; j++) {
                    temps->en_temp[j] = weight * fabs(opt->population[indices[soa->size_reg[r] - 1]].position[j] - 
                                                      opt->population[indices[rand_en]].position[j]);
                }
            }

            // Process solutions
            for (int s = soa->start_reg[r]; s < soa->end_reg[r]; s++) {
                // Compute mu_s
                __m128d mu_vec = _mm_set1_pd(mu);
                __m128d one_minus_mu = _mm_set1_pd(1.0 - mu);
                if (opt->dim == 2) {
                    double r1 = xorshift_random(&temps->rng);
                    double r2 = xorshift_random(&temps->rng);
                    __m128d rand_vec = _mm_set_pd(r2, r1);
                    __m128d mu_s = _mm_add_pd(mu_vec, _mm_mul_pd(one_minus_mu, rand_vec));
                    __m128d zero = _mm_setzero_pd();
                    __m128d one = _mm_set1_pd(1.0);
                    __m128d mask = _mm_or_pd(_mm_cmple_pd(mu_s, zero), _mm_cmpge_pd(mu_s, one));
                    mu_s = _mm_or_pd(_mm_andnot_pd(mask, mu_s), _mm_and_pd(mask, mu_vec));
                    _mm_storeu_pd(temps->mu_s, mu_s);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        temps->mu_s[j] = mu + (1.0 - mu) * xorshift_random(&temps->rng);
                        if (temps->mu_s[j] <= 0.0 || temps->mu_s[j] >= 1.0) temps->mu_s[j] = mu;
                    }
                }

                // Compute directions (SIMD for dim=2)
                if (opt->dim == 2) {
                    __m128d pos = _mm_loadu_pd(opt->population[s].position);
                    __m128d pbest = _mm_loadu_pd(soa->pbest_s + s * opt->dim);
                    __m128d lbest = _mm_loadu_pd(soa->lbest_s + r * opt->dim);
                    __m128d worst = _mm_loadu_pd(opt->population[indices[soa->size_reg[r] - 1]].position);
                    __m128d x_pdirect = _mm_sub_pd(pbest, pos);
                    __m128d x_ldirect1 = (soa->lbest_fun[r] < opt->population[s].fitness) ? 
                                        _mm_sub_pd(lbest, pos) : _mm_setzero_pd();
                    __m128d x_ldirect2 = (opt->population[indices[soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 
                                        _mm_sub_pd(worst, pos) : _mm_setzero_pd();
                    _mm_storeu_pd(temps->x_pdirect, x_pdirect);
                    _mm_storeu_pd(temps->x_ldirect1, x_ldirect1);
                    _mm_storeu_pd(temps->x_ldirect2, x_ldirect2);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        temps->x_pdirect[j] = soa->pbest_s[s * opt->dim + j] - opt->population[s].position[j];
                        temps->x_ldirect1[j] = (soa->lbest_fun[r] < opt->population[s].fitness) ? 
                                              soa->lbest_s[r * opt->dim + j] - opt->population[s].position[j] : 0.0;
                        temps->x_ldirect2[j] = (opt->population[indices[soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 
                                              opt->population[indices[soa->size_reg[r] - 1]].position[j] - opt->population[s].position[j] : 0.0;
                    }
                }

                // Temporal direction
                double f_values[3] = {soa->f_t_2[s], soa->f_t_1[s], opt->population[s].fitness};
                double *e_values[3] = {soa->e_t_2 + s * opt->dim, soa->e_t_1 + s * opt->dim, opt->population[s].position};
                int order_idx[3] = {0, 1, 2};
                if (f_values[0] > f_values[1]) { int t = order_idx[0]; order_idx[0] = order_idx[1]; order_idx[1] = t; }
                if (f_values[order_idx[1]] > f_values[2]) { int t = order_idx[1]; order_idx[1] = order_idx[2]; order_idx[2] = t; }
                if (f_values[order_idx[0]] > f_values[order_idx[1]]) { int t = order_idx[0]; order_idx[0] = order_idx[1]; order_idx[1] = t; }
                if (opt->dim == 2) {
                    __m128d e_best = _mm_loadu_pd(e_values[order_idx[0]]);
                    __m128d e_worst = _mm_loadu_pd(e_values[order_idx[2]]);
                    __m128d x_tdirect = _mm_sub_pd(e_best, e_worst);
                    _mm_storeu_pd(temps->x_tdirect, x_tdirect);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        temps->x_tdirect[j] = e_values[order_idx[0]][j] - e_values[order_idx[2]][j];
                    }
                }

                // Compute direction signs
                int flag_direct[4] = {1, 1, (soa->lbest_fun[r] < opt->population[s].fitness) ? 1 : 0, 
                                     (opt->population[indices[soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 1 : 0};
                double x_signs[4][2];
                if (opt->dim == 2) {
                    __m128d zero = _mm_setzero_pd();
                    __m128d one = _mm_set1_pd(1.0);
                    __m128d neg_one = _mm_set1_pd(-1.0);
                    for (int i = 0; i < 4; i++) {
                        __m128d dir = _mm_loadu_pd((i == 0) ? temps->x_tdirect : 
                                                  (i == 1) ? temps->x_pdirect : 
                                                  (i == 2) ? temps->x_ldirect1 : temps->x_ldirect2);
                        __m128d gt_zero = _mm_cmpgt_pd(dir, zero);
                        __m128d lt_zero = _mm_cmplt_pd(dir, zero);
                        __m128d sign = _mm_or_pd(_mm_and_pd(gt_zero, one), _mm_and_pd(lt_zero, neg_one));
                        _mm_storeu_pd(x_signs[i], sign);
                    }
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        x_signs[0][j] = (temps->x_tdirect[j] > 0) ? 1.0 : (temps->x_tdirect[j] < 0) ? -1.0 : 0.0;
                        x_signs[1][j] = (temps->x_pdirect[j] > 0) ? 1.0 : (temps->x_pdirect[j] < 0) ? -1.0 : 0.0;
                        x_signs[2][j] = (temps->x_ldirect1[j] > 0) ? 1.0 : (temps->x_ldirect1[j] < 0) ? -1.0 : 0.0;
                        x_signs[3][j] = (temps->x_ldirect2[j] > 0) ? 1.0 : (temps->x_ldirect2[j] < 0) ? -1.0 : 0.0;
                    }
                }

                int select_sign[4], num_sign = 0;
                for (int i = 0; i < 4; i++) {
                    if (flag_direct[i]) select_sign[num_sign++] = i;
                }

                memset(temps->num_pone, 0, opt->dim * sizeof(double));
                memset(temps->num_none, 0, opt->dim * sizeof(double));
                for (int i = 0; i < num_sign; i++) {
                    for (int j = 0; j < opt->dim; j++) {
                        temps->num_pone[j] += (fabs(x_signs[select_sign[i]][j]) + x_signs[select_sign[i]][j]) / 2.0;
                        temps->num_none[j] += (fabs(x_signs[select_sign[i]][j]) - x_signs[select_sign[i]][j]) / 2.0;
                    }
                }

                // Compute x_direct (SIMD for dim=2)
                if (opt->dim == 2) {
                    __m128d prob_pone = _mm_set1_pd(num_sign > 0 ? temps->num_pone[0] / num_sign : 0.5);
                    __m128d prob_none = _mm_set1_pd(num_sign > 0 ? (temps->num_pone[0] + temps->num_none[0]) / num_sign : 1.0);
                    double r1 = xorshift_random(&temps->rng);
                    __m128d rand_vec = _mm_set1_pd(r1);
                    __m128d one = _mm_set1_pd(1.0);
                    __m128d neg_one = _mm_set1_pd(-1.0);
                    __m128d zero = _mm_setzero_pd();
                    __m128d le_pone = _mm_cmple_pd(rand_vec, prob_pone);
                    __m128d le_none = _mm_cmple_pd(rand_vec, prob_none);
                    __m128d x_direct = _mm_or_pd(_mm_and_pd(le_pone, one), 
                                                _mm_and_pd(_mm_andnot_pd(le_pone, le_none), neg_one));
                    _mm_storeu_pd(temps->x_direct, x_direct);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        double prob_pone = num_sign > 0 ? temps->num_pone[j] / num_sign : 0.5;
                        double prob_none = num_sign > 0 ? (temps->num_pone[j] + temps->num_none[j]) / num_sign : 1.0;
                        double r = xorshift_random(&temps->rng);
                        temps->x_direct[j] = (r <= prob_pone) ? 1.0 : (r <= prob_none) ? -1.0 : 0.0;
                    }
                }

                // Adjust direction based on bounds
                if (opt->dim == 2) {
                    __m128d pos = _mm_loadu_pd(opt->population[s].position);
                    __m128d bound_min = _mm_set_pd(opt->bounds[2], opt->bounds[0]);
                    __m128d bound_max = _mm_set_pd(opt->bounds[3], opt->bounds[1]);
                    __m128d lt_min = _mm_cmplt_pd(pos, bound_min);
                    __m128d gt_max = _mm_cmpgt_pd(pos, bound_max);
                    __m128d x_direct = _mm_loadu_pd(temps->x_direct);
                    x_direct = _mm_or_pd(_mm_and_pd(lt_min, _mm_set1_pd(1.0)), 
                                        _mm_andnot_pd(lt_min, x_direct));
                    x_direct = _mm_or_pd(_mm_and_pd(gt_max, _mm_set1_pd(-1.0)), 
                                        _mm_andnot_pd(gt_max, x_direct));
                    __m128d zero = _mm_setzero_pd();
                    __m128d is_zero = _mm_cmpeq_pd(x_direct, zero);
                    double r = xorshift_random(&temps->rng);
                    __m128d rand_sign = _mm_set1_pd(r < 0.5 ? 1.0 : -1.0);
                    x_direct = _mm_or_pd(_mm_andnot_pd(is_zero, x_direct), 
                                        _mm_and_pd(is_zero, rand_sign));
                    _mm_storeu_pd(temps->x_direct, x_direct);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        if (opt->population[s].position[j] > opt->bounds[2 * j + 1]) temps->x_direct[j] = -1.0;
                        if (opt->population[s].position[j] < opt->bounds[2 * j]) temps->x_direct[j] = 1.0;
                        if (temps->x_direct[j] == 0.0) {
                            temps->x_direct[j] = xorshift_random(&temps->rng) < 0.5 ? 1.0 : -1.0;
                        }
                    }
                }

                // Compute step (SIMD for dim=2)
                if (opt->dim == 2) {
                    __m128d x_direct = _mm_loadu_pd(temps->x_direct);
                    __m128d en_temp = _mm_loadu_pd(temps->en_temp);
                    __m128d mu_s = _mm_loadu_pd(temps->mu_s);
                    __m128d neg_two = _mm_set1_pd(-2.0);
                    __m128d log_mu = _mm_set1_pd(log(mu_s[0]));
                    __m128d sqrt_term = _mm_sqrt_pd(_mm_mul_pd(neg_two, log_mu));
                    __m128d r_temp = _mm_mul_pd(x_direct, _mm_mul_pd(en_temp, sqrt_term));
                    __m128d rmin = _mm_loadu_pd(soa->rmin);
                    __m128d rmax = _mm_loadu_pd(soa->rmax);
                    r_temp = _mm_max_pd(rmin, _mm_min_pd(rmax, r_temp));
                    __m128d pos = _mm_loadu_pd(opt->population[s].position);
                    __m128d new_pos = _mm_add_pd(pos, r_temp);
                    _mm_storeu_pd(temps->temp_population + s * opt->dim, new_pos);
                } else {
                    for (int j = 0; j < opt->dim; j++) {
                        temps->r_temp[j] = temps->x_direct[j] * (temps->en_temp[j] * sqrt(-2.0 * log(temps->mu_s[j])));
                        temps->r_temp[j] = fmax(soa->rmin[j], fmin(soa->rmax[j], temps->r_temp[j]));
                        temps->temp_population[s * opt->dim + j] = opt->population[s].position[j] + temps->r_temp[j];
                    }
                }
            }
            idx_offset += soa->size_reg[r];
        }

        // Evaluate offspring
        for (int s = 0; s < opt->population_size; s++) {
            double temp_val = objective_function(temps->temp_population + s * opt->dim);
            fes++;
            temps->indices_changed[s / (opt->population_size / soa->num_regions)] = 1;

            memcpy(soa->e_t_2 + s * opt->dim, soa->e_t_1 + s * opt->dim, opt->dim * sizeof(double));
            memcpy(soa->e_t_1 + s * opt->dim, opt->population[s].position, opt->dim * sizeof(double));
            memcpy(opt->population[s].position, temps->temp_population + s * opt->dim, opt->dim * sizeof(double));
            soa->f_t_2[s] = soa->f_t_1[s];
            soa->f_t_1[s] = opt->population[s].fitness;
            opt->population[s].fitness = temp_val;

            if (temp_val < soa->pbest_fun[s]) {
                soa->pbest_fun[s] = temp_val;
                memcpy(soa->pbest_s + s * opt->dim, opt->population[s].position, opt->dim * sizeof(double));
            }
            if (temp_val < opt->best_solution.fitness) {
                opt->best_solution.fitness = temp_val;
                memcpy(opt->best_solution.position, opt->population[s].position, opt->dim * sizeof(double));
            }
        }

        // Shuffle population if no improvement
        if (error_prev <= opt->best_solution.fitness || gens % 1 == 0) {
            int *perm = temps->sorted_indices;
            for (int i = 0; i < opt->population_size; i++) perm[i] = i;
            for (int i = opt->population_size - 1; i > 0; i--) {
                int j = (int)(xorshift_random(&temps->rng) * (i + 1));
                int temp = perm[i];
                perm[i] = perm[j];
                perm[j] = temp;
            }

            double *temp_pop = temps->temp_population;
            for (int i = 0; i < opt->population_size; i++) {
                int idx = perm[i];
                memcpy(temp_pop + i * opt->dim, opt->population[idx].position, opt->dim * sizeof(double));
                memcpy(soa->pbest_s + i * opt->dim, soa->pbest_s + idx * opt->dim, opt->dim * sizeof(double));
                memcpy(soa->e_t_1 + i * opt->dim, soa->e_t_1 + idx * opt->dim, opt->dim * sizeof(double));
                memcpy(soa->e_t_2 + i * opt->dim, soa->e_t_2 + idx * opt->dim, opt->dim * sizeof(double));
                soa->pbest_fun[i] = soa->pbest_fun[idx];
                soa->f_t_1[i] = soa->f_t_1[idx];
                soa->f_t_2[i] = soa->f_t_2[idx];
                opt->population[i].fitness = opt->population[idx].fitness;
            }

            for (int i = 0; i < opt->population_size; i++) {
                memcpy(opt->population[i].position, temp_pop + i * opt->dim, opt->dim * sizeof(double));
            }
            for (int r = 0; r < soa->num_regions; r++) temps->indices_changed[r] = 1;
        }

        error_prev = opt->best_solution.fitness;

        // Inline update_region_bests
        for (int r = 0; r < soa->num_regions; r++) {
            if (soa->size_reg[r] <= 0) continue;
            int best_idx = soa->start_reg[r];
            double best_fun = opt->population[best_idx].fitness;
            for (int i = soa->start_reg[r]; i < soa->end_reg[r]; i++) {
                if (opt->population[i].fitness < best_fun) {
                    best_fun = opt->population[i].fitness;
                    best_idx = i;
                }
            }
            if (best_fun < soa->lbest_fun[r]) {
                soa->lbest_fun[r] = best_fun;
                memcpy(soa->lbest_s + r * opt->dim, opt->population[best_idx].position, opt->dim * sizeof(double));
            }
        }
        enforce_bound_constraints(opt);
    }

    free_temps(temps);
    DEBUG_LOG("SOA_optimize: Optimization completed\n");
    if (soa->magic != SOA_MAGIC) {
        DEBUG_LOG("SOA_optimize: Freeing temporary SOA at %p\n", soa);
        SOA_free(soa);
    }
}
