#include "TFWO.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// Fast Xorshift RNG
static uint64_t rng_state = 0x123456789abcdef0ULL;
inline void seed_rng(uint64_t seed) {
    rng_state = seed ? seed : 0x123456789abcdef0ULL;
}

inline double tfwo_rand_double(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((rng_state * 0x2545F4914F6CDD1DULL) >> 32) / (double)(1ULL << 32);
}

// Fast approximations for math functions
inline double fast_cos(double x) {
    x = x - (int)(x / (2.0 * PI)) * 2.0 * PI;
    if (x < 0.0) x += 2.0 * PI;
    double x2 = x * x;
    return 1.0 - x2 * 0.5 + x2 * x2 * 0.041666666666666664;
}

inline double fast_sin(double x) {
    return fast_cos(x - PI * 0.5);
}

inline double fast_sqrt(double x) {
    if (x <= 0.0) return 0.0;
    double y = _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)));
    return y;
}

// Enforce bounds using SIMD
inline void enforce_bounds(double *position, double *bounds, int dim) {
    int i = 0;
    for (; i <= dim - 4; i += 4) {
        __m256d pos = _mm256_load_pd(position + i);
        __m256d lower = _mm256_loadu_pd(bounds + 2 * i);
        __m256d upper = _mm256_loadu_pd(bounds + 2 * i + 1);
        pos = _mm256_max_pd(pos, lower);
        pos = _mm256_min_pd(pos, upper);
        _mm256_store_pd(position + i, pos);
    }
    for (; i < dim; i++) {
        position[i] = fmax(bounds[2 * i], fmin(bounds[2 * i + 1], position[i]));
    }
}

// Structure for sorting
typedef struct {
    int idx;
    double cost;
} SortEntry;

// Comparison function for qsort
static int compare_objects(const void *a, const void *b) {
    double cost_a = ((SortEntry *)a)->cost;
    double cost_b = ((SortEntry *)b)->cost;
    return (cost_a > cost_b) - (cost_a < cost_b);
}

// Initialize whirlpools and objects
void initialize_whirlpools(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in initialize_whirlpools\n");
        return;
    }

    int dim = data->dim = opt->dim;
    int n_pop = data->n_whirlpools + data->n_whirlpools * data->n_objects_per_whirlpool;
    data->total_positions = n_pop;

    // Allocate contiguous memory
    data->positions = (double *)_mm_malloc(n_pop * dim * sizeof(double), ALIGNMENT);
    data->costs = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    data->deltas = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    data->position_sums = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    if (!data->positions || !data->costs || !data->deltas || !data->position_sums) {
        fprintf(stderr, "Error: Failed to allocate arrays\n");
        _mm_free(data->positions);
        _mm_free(data->costs);
        _mm_free(data->deltas);
        _mm_free(data->position_sums);
        return;
    }

    // Initialize positions and costs
    for (int i = 0; i < n_pop; i++) {
        double *pos = data->positions + i * dim;
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
            double min = opt->bounds[2 * j];
            double max = opt->bounds[2 * j + 1];
            pos[j] = tfwo_rand_double() * (max - min) + min;
            sum += pos[j];
        }
        data->position_sums[i] = sum;
        data->costs[i] = objective_function(pos);
        data->deltas[i] = 0.0;
    }

    // Sort by cost using qsort
    SortEntry *entries = (SortEntry *)malloc(n_pop * sizeof(SortEntry));
    if (!entries) {
        fprintf(stderr, "Error: Failed to allocate sort entries\n");
        _mm_free(data->positions);
        _mm_free(data->costs);
        _mm_free(data->deltas);
        _mm_free(data->position_sums);
        return;
    }
    for (int i = 0; i < n_pop; i++) {
        entries[i].idx = i;
        entries[i].cost = data->costs[i];
    }
    qsort(entries, n_pop, sizeof(SortEntry), compare_objects);

    // Reorganize data
    double *new_positions = (double *)_mm_malloc(n_pop * dim * sizeof(double), ALIGNMENT);
    double *new_costs = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    double *new_deltas = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    double *new_sums = (double *)_mm_malloc(n_pop * sizeof(double), ALIGNMENT);
    if (!new_positions || !new_costs || !new_deltas || !new_sums) {
        fprintf(stderr, "Error: Failed to allocate new arrays\n");
        _mm_free(new_positions);
        _mm_free(new_costs);
        _mm_free(new_deltas);
        _mm_free(new_sums);
        free(entries);
        _mm_free(data->positions);
        _mm_free(data->costs);
        _mm_free(data->deltas);
        _mm_free(data->position_sums);
        return;
    }
    for (int i = 0; i < n_pop; i++) {
        int idx = entries[i].idx;
        for (int j = 0; j < dim; j++) {
            new_positions[i * dim + j] = data->positions[idx * dim + j];
        }
        new_costs[i] = data->costs[idx];
        new_deltas[i] = data->deltas[idx];
        new_sums[i] = data->position_sums[idx];
    }
    _mm_free(data->positions);
    _mm_free(data->costs);
    _mm_free(data->deltas);
    _mm_free(data->position_sums);
    data->positions = new_positions;
    data->costs = new_costs;
    data->deltas = new_deltas;
    data->position_sums = new_sums;
    free(entries);
}

// Update objects and whirlpool positions
void effects_of_whirlpools(Optimizer *opt, TFWO_Data *data, int iter, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in effects_of_whirlpools\n");
        return;
    }

    int dim = data->dim;
    int n_whirlpools = data->n_whirlpools;
    int n_objs = data->n_objects_per_whirlpool;
    double *positions = data->positions;
    double *costs = data->costs;
    double *deltas = data->deltas;
    double *sums = data->position_sums;
    double *d = data->temp_d;
    double *d2 = data->temp_d2;
    double *RR = data->temp_RR;
    double *J = data->temp_J;

    for (int i = 0; i < n_whirlpools; i++) {
        int wp_idx = i;
        double *wp_pos = positions + wp_idx * dim;
        double wp_cost = costs[wp_idx];
        double wp_sum = sums[wp_idx];
        int obj_start = n_whirlpools + i * n_objs;

        // Update objects
        for (int j = 0; j < n_objs; j++) {
            int obj_idx = obj_start + j;
            double *obj_pos = positions + obj_idx * dim;
            double obj_cost = costs[obj_idx];
            double obj_delta = deltas[obj_idx];
            double obj_sum = sums[obj_idx];

            int min_idx = wp_idx, max_idx = wp_idx;
            if (n_whirlpools > 1) {
                int J_idx = 0;
                for (int t = 0; t < n_whirlpools; t++) {
                    if (t != wp_idx) {
                        J[J_idx] = fast_sqrt(fabs(sums[t] - obj_sum)) * fabs(costs[t]);
                        J_idx++;
                    }
                }
                min_idx = 0;
                max_idx = 0;
                for (int t = 1; t < n_whirlpools - 1; t++) {
                    if (J[t] < J[min_idx]) min_idx = t;
                    if (J[t] > J[max_idx]) max_idx = t;
                }
                if (min_idx >= wp_idx) min_idx++;
                if (max_idx >= wp_idx) max_idx++;
                double *min_pos = positions + min_idx * dim;
                double *max_pos = positions + max_idx * dim;
                __m256d r1 = _mm256_set1_pd(tfwo_rand_double());
                __m256d r2 = _mm256_set1_pd(tfwo_rand_double());
                for (int k = 0; k <= dim - 4; k += 4) {
                    __m256d min_p = _mm256_load_pd(min_pos + k);
                    __m256d max_p = _mm256_load_pd(max_pos + k);
                    __m256d obj_p = _mm256_load_pd(obj_pos + k);
                    __m256d diff1 = _mm256_sub_pd(min_p, obj_p);
                    __m256d diff2 = _mm256_sub_pd(max_p, obj_p);
                    _mm256_store_pd(d + k, _mm256_mul_pd(r1, diff1));
                    _mm256_store_pd(d2 + k, _mm256_mul_pd(r2, diff2));
                }
                for (int k = (dim / 4) * 4; k < dim; k++) {
                    d[k] = tfwo_rand_double() * (min_pos[k] - obj_pos[k]);
                    d2[k] = tfwo_rand_double() * (max_pos[k] - obj_pos[k]);
                }
            } else {
                __m256d r = _mm256_set1_pd(tfwo_rand_double());
                for (int k = 0; k <= dim - 4; k += 4) {
                    __m256d wp_p = _mm256_load_pd(wp_pos + k);
                    __m256d obj_p = _mm256_load_pd(obj_pos + k);
                    __m256d diff = _mm256_sub_pd(wp_p, obj_p);
                    _mm256_store_pd(d + k, _mm256_mul_pd(r, diff));
                    _mm256_store_pd(d2 + k, _mm256_setzero_pd());
                }
                for (int k = (dim / 4) * 4; k < dim; k++) {
                    d[k] = tfwo_rand_double() * (wp_pos[k] - obj_pos[k]);
                    d2[k] = 0.0;
                }
            }

            // Update delta and compute trigonometrics
            obj_delta += tfwo_rand_double() * tfwo_rand_double() * PI;
            double cos_eee = fast_cos(obj_delta);
            double sin_eee = fast_sin(obj_delta);
            double fr0 = cos_eee;
            double fr10 = -sin_eee;
            double fr0_fr10 = fr0 * fr10;
            __m256d fr0_v = _mm256_set1_pd(fr0);
            __m256d fr10_v = _mm256_set1_pd(fr10);
            __m256d scale = _mm256_set1_pd(1.0 + fr0_fr10);

            // Compute new position
            double sum = 0.0;
            for (int k = 0; k <= dim - 4; k += 4) {
                __m256d d_v = _mm256_load_pd(d + k);
                __m256d d2_v = _mm256_load_pd(d2 + k);
                __m256d wp_p = _mm256_load_pd(wp_pos + k);
                __m256d x = _mm256_add_pd(_mm256_mul_pd(fr0_v, d_v), _mm256_mul_pd(fr10_v, d2_v));
                x = _mm256_mul_pd(x, scale);
                __m256d new_p = _mm256_sub_pd(wp_p, x);
                _mm256_store_pd(RR + k, new_p);
            }
            for (int k = (dim / 4) * 4; k < dim; k++) {
                double x = (fr0 * d[k] + fr10 * d2[k]) * (1.0 + fr0_fr10);
                RR[k] = wp_pos[k] - x;
                sum += RR[k];
            }
            enforce_bounds(RR, opt->bounds, dim);
            double cost = objective_function(RR);

            // Update if better
            if (cost <= obj_cost) {
                costs[obj_idx] = cost;
                for (int k = 0; k <= dim - 4; k += 4) {
                    __m256d new_p = _mm256_load_pd(RR + k);
                    _mm256_store_pd(obj_pos + k, new_p);
                }
                for (int k = (dim / 4) * 4; k < dim; k++) {
                    obj_pos[k] = RR[k];
                }
                sums[obj_idx] = sum;
            }

            // Random jump
            double cos_eee_sq = cos_eee * cos_eee;
            double sin_eee_sq = sin_eee * sin_eee;
            double FE_i = cos_eee_sq * sin_eee_sq * cos_eee_sq * sin_eee_sq;
            if (tfwo_rand_double() < FE_i) {
                int k = (int)(tfwo_rand_double() * dim);
                double min = opt->bounds[2 * k];
                double max = opt->bounds[2 * k + 1];
                obj_pos[k] = tfwo_rand_double() * (max - min) + min;
                costs[obj_idx] = objective_function(obj_pos);
                sum = 0.0;
                for (int m = 0; m < dim; m++) sum += obj_pos[m];
                sums[obj_idx] = sum;
            }
            deltas[obj_idx] = obj_delta;
        }

        // Update whirlpool position
        for (int t = 0; t < n_whirlpools; t++) {
            J[t] = (t == wp_idx) ? INFINITY : costs[t] * fabs(sums[t] - wp_sum);
        }
        int min_idx = 0;
        for (int t = 1; t < n_whirlpools; t++) {
            if (J[t] < J[min_idx]) min_idx = t;
        }
        deltas[wp_idx] += tfwo_rand_double() * tfwo_rand_double() * PI;
        double fr = fast_cos(deltas[wp_idx]) + fast_sin(deltas[wp_idx]);
        if (fr < 0.0) fr = -fr;
        double *min_pos = positions + min_idx * dim;
        double sum = 0.0;
        __m256d fr_v = _mm256_set1_pd(fr);
        __m256d r_v = _mm256_set1_pd(tfwo_rand_double());
        for (int k = 0; k <= dim - 4; k += 4) {
            __m256d min_p = _mm256_load_pd(min_pos + k);
            __m256d wp_p = _mm256_load_pd(wp_pos + k);
            __m256d diff = _mm256_sub_pd(min_p, wp_p);
            __m256d x = _mm256_mul_pd(fr_v, _mm256_mul_pd(r_v, diff));
            __m256d new_p = _mm256_sub_pd(min_p, x);
            _mm256_store_pd(RR + k, new_p);
        }
        for (int k = (dim / 4) * 4; k < dim; k++) {
            double x = fr * tfwo_rand_double() * (min_pos[k] - wp_pos[k]);
            RR[k] = min_pos[k] - x;
            sum += RR[k];
        }
        enforce_bounds(RR, opt->bounds, dim);
        double new_cost = objective_function(RR);
        if (new_cost <= wp_cost) {
            costs[wp_idx] = new_cost;
            for (int k = 0; k <= dim - 4; k += 4) {
                __m256d new_p = _mm256_load_pd(RR + k);
                _mm256_store_pd(wp_pos + k, new_p);
            }
            for (int k = (dim / 4) * 4; k < dim; k++) {
                wp_pos[k] = RR[k];
            }
            sums[wp_idx] = sum;
        }
    }
}

// Update whirlpool with best object if better
void update_best_whirlpool(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in update_best_whirlpool\n");
        return;
    }

    int dim = data->dim;
    int n_whirlpools = data->n_whirlpools;
    int n_objs = data->n_objects_per_whirlpool;
    double *positions = data->positions;
    double *costs = data->costs;
    double *sums = data->position_sums;

    for (int i = 0; i < n_whirlpools; i++) {
        int wp_idx = i;
        double wp_cost = costs[wp_idx];
        double wp_sum = sums[wp_idx];
        double *wp_pos = positions + wp_idx * dim;
        int obj_start = n_whirlpools + i * n_objs;

        double min_cost = costs[obj_start];
        int min_cost_idx = obj_start;
        for (int j = 1; j < n_objs; j++) {
            int obj_idx = obj_start + j;
            if (costs[obj_idx] < min_cost) {
                min_cost = costs[obj_idx];
                min_cost_idx = obj_idx;
            }
        }
        if (min_cost <= wp_cost) {
            double *obj_pos = positions + min_cost_idx * dim;
            double sum = sums[min_cost_idx];
            for (int k = 0; k <= dim - 4; k += 4) {
                __m256d wp_p = _mm256_load_pd(wp_pos + k);
                __m256d obj_p = _mm256_load_pd(obj_pos + k);
                _mm256_store_pd(wp_pos + k, obj_p);
                _mm256_store_pd(obj_pos + k, wp_p);
            }
            for (int k = (dim / 4) * 4; k < dim; k++) {
                double temp = wp_pos[k];
                wp_pos[k] = obj_pos[k];
                obj_pos[k] = temp;
            }
            sums[wp_idx] = sum;
            sums[min_cost_idx] = wp_sum;
            costs[min_cost_idx] = wp_cost;
            costs[wp_idx] = min_cost;
        }
    }
}

// Free TFWO data
void free_tfwo_data(TFWO_Data *data) {
    if (!data) return;
    _mm_free(data->positions);
    _mm_free(data->costs);
    _mm_free(data->deltas);
    _mm_free(data->position_sums);
    _mm_free(data->best_costs);
    _mm_free(data->mean_costs);
    _mm_free(data->temp_d);
    _mm_free(data->temp_d2);
    _mm_free(data->temp_RR);
    _mm_free(data->temp_J);
    free(data);
}

// Main optimization function
void TFWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Null pointer in TFWO_optimize\n");
        return;
    }

    seed_rng((uint64_t)time(NULL));

    TFWO_Data *data = (TFWO_Data *)calloc(1, sizeof(TFWO_Data));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate TFWO_Data\n");
        return;
    }
    data->n_whirlpools = N_WHIRLPOOLS_DEFAULT;
    data->n_objects_per_whirlpool = N_OBJECTS_PER_WHIRLPOOL_DEFAULT;
    data->best_costs = (double *)_mm_malloc(opt->max_iter * sizeof(double), ALIGNMENT);
    data->mean_costs = (double *)_mm_malloc(opt->max_iter * sizeof(double), ALIGNMENT);
    data->temp_d = (double *)_mm_malloc(opt->dim * sizeof(double), ALIGNMENT);
    data->temp_d2 = (double *)_mm_malloc(opt->dim * sizeof(double), ALIGNMENT);
    data->temp_RR = (double *)_mm_malloc(opt->dim * sizeof(double), ALIGNMENT);
    data->temp_J = (double *)_mm_malloc(data->n_whirlpools * sizeof(double), ALIGNMENT);
    if (!data->best_costs || !data->mean_costs || !data->temp_d || !data->temp_d2 || !data->temp_RR || !data->temp_J) {
        fprintf(stderr, "Error: Failed to allocate arrays\n");
        _mm_free(data->best_costs);
        _mm_free(data->mean_costs);
        _mm_free(data->temp_d);
        _mm_free(data->temp_d2);
        _mm_free(data->temp_RR);
        _mm_free(data->temp_J);
        free(data);
        return;
    }

    initialize_whirlpools(opt, data, objective_function);
    if (!data->positions) {
        fprintf(stderr, "Error: Whirlpool initialization failed\n");
        free_tfwo_data(data);
        return;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        effects_of_whirlpools(opt, data, iter, objective_function);
        update_best_whirlpool(opt, data, objective_function);

        double best_cost = data->costs[0];
        int best_idx = 0;
        double mean_cost = best_cost;
        for (int i = 1; i < data->n_whirlpools; i++) {
            double cost = data->costs[i];
            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
            mean_cost += cost;
        }
        mean_cost /= data->n_whirlpools;

        if (best_cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_cost;
            double *dst = opt->best_solution.position;
            double *src = data->positions + best_idx * data->dim;
            for (int k = 0; k <= data->dim - 4; k += 4) {
                __m256d s = _mm256_load_pd(src + k);
                _mm256_storeu_pd(dst + k, s);
            }
            for (int k = (data->dim / 4) * 4; k < data->dim; k++) {
                dst[k] = src[k];
            }
        }

        data->best_costs[iter] = best_cost;
        data->mean_costs[iter] = mean_cost;

        printf("Iter %d: Best Cost = %f\n", iter + 1, best_cost);
    }

    free_tfwo_data(data);
}
