#include "WCA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Enforce bounds with SIMD
void wca_enforce_bound_constraints(Optimizer *opt) {
    #ifdef __AVX2__
    __m256d min_vec, max_vec, pos_vec;
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j += 4) {
            if (j + 4 <= opt->dim) {
                min_vec = _mm256_set1_pd(opt->bounds[2 * j]);
                max_vec = _mm256_set1_pd(opt->bounds[2 * j + 1]);
                pos_vec = _mm256_loadu_pd(&opt->population[i].position[j]);
                pos_vec = _mm256_max_pd(min_vec, _mm256_min_pd(max_vec, pos_vec));
                _mm256_storeu_pd(&opt->population[i].position[j], pos_vec);
            } else {
                #pragma unroll
                for (; j < opt->dim; j++) {
                    if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                        opt->population[i].position[j] = opt->bounds[2 * j];
                    } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                        opt->population[i].position[j] = opt->bounds[2 * j + 1];
                    }
                }
            }
        }
    }
    #else
    for (int i = 0; i < opt->population_size; i++) {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            } else if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
    #endif
}

// Initialize streams
void initialize_streams(Optimizer *opt) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < opt->population_size; i++) {
        #pragma omp simd
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double_wca(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

// Evaluate streams with OpenMP
void evaluate_streams(Optimizer *opt, double (*objective_function)(double *)) {
    double prev_best = opt->best_solution.fitness;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = objective_function(opt->population[i].position);
        opt->population[i].fitness = fitness;
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            if (fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = fitness;
                #pragma omp simd
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }
    if (fabs(prev_best - opt->best_solution.fitness) < CONVERGENCE_THRESHOLD) {
        opt->max_iter = 0;
    }
}

// Simplified IOP
double iop(double *C, double *D, int rows, int cols_C, int cols_D, int *cache) {
    if (cols_C == 0 || rows == 0) return 0.0;

    double sum_diff = 0.0;
    int *counts_C = cache;
    int *counts_CD = cache + rows;

    #pragma omp parallel for reduction(+:sum_diff)
    for (int i = 0; i < rows; i++) {
        counts_C[i] = 1;
        counts_CD[i] = 1;
        sum_diff += (double)(counts_C[i] - counts_CD[i]);
    }
    return sum_diff;
}

// Positive region with SIMD
int* positive_region(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    if (cols_C == 0 || rows == 0) {
        *size = 0;
        return NULL;
    }

    int *pos = (int *)_mm_malloc(rows * sizeof(int), 32);
    if (!pos) return NULL;
    *size = 0;
    double diff = iop(C, D, rows, cols_C, cols_D, cache);
    if (fabs(diff) < 1e-6) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            pos[i] = i;
        }
        *size = rows;
    }
    return pos;
}

// Quick sort with OpenMP tasks
void quick_sort(double *arr, int *indices, int low, int high) {
    if (low < high) {
        double pivot = arr[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[indices[j]] <= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;

        #ifdef _OPENMP
        #pragma omp task
        #endif
        quick_sort(arr, indices, low, i);
        #ifdef _OPENMP
        #pragma omp task
        #endif
        quick_sort(arr, indices, i + 2, high);
    }
}

// Fast reduction with incremental IOP
int* fast_red(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    double iop_C = iop(C, D, rows, cols_C, cols_D, cache);
    double *w = (double *)_mm_malloc(cols_C * sizeof(double), 32);
    int *ind = (int *)_mm_malloc(cols_C * sizeof(int), 32);
    int red[WCA_MAX_DIM];
    *size = 0;

    double *C_single = (double *)_mm_malloc(rows * sizeof(double), 32);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < cols_C; i++) {
        #pragma omp simd
        for (int r = 0; r < rows; r++) {
            C_single[r] = C[r * cols_C + i];
        }
        w[i] = iop(C_single, D, rows, 1, cols_D, cache);
        ind[i] = i;
    }
    _mm_free(C_single);

    #ifdef _OPENMP
    #pragma omp parallel
    #pragma omp single
    #endif
    quick_sort(w, ind, 0, cols_C - 1);

    double *C_red = (double *)_mm_malloc(rows * cols_C * sizeof(double), 32);
    for (int i = 0; i < cols_C; i++) {
        red[(*size)++] = ind[i];
        #pragma omp parallel for
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < *size; j++) {
                C_red[r * cols_C + j] = C[r * cols_C + red[j]];
            }
        }
        if (fabs(iop(C_red, D, rows, *size, cols_D, cache) - iop_C) < EARLY_STOP_THRESHOLD) {
            break;
        }
    }

    int *result = (int *)_mm_malloc(*size * sizeof(int), 32);
    memcpy(result, red, *size * sizeof(int));
    _mm_free(w);
    _mm_free(ind);
    _mm_free(C_red);
    return result;
}

// APS mechanism with bitset
void aps_mechanism(double *C, double *D, int rows, int cols_C, int cols_D, int *B, int B_size, int *u, int *v, int *cache) {
    *u = 0;
    *v = 0;
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size, cache);
    double *C_B = (double *)_mm_malloc(rows * B_size * sizeof(double), 32);
    #pragma omp parallel for
    for (int r = 0; r < rows; r++) {
        for (int j = 0; j < B_size; j++) {
            C_B[r * B_size + j] = C[r * cols_C + B[j]];
        }
    }
    int unpos_size;
    int *unpos = positive_region(C_B, D, rows, B_size, cols_D, &unpos_size, cache);
    _mm_free(C_B);

    int unred[WCA_MAX_DIM];
    int unred_size = 0;
    #pragma unroll
    for (int i = 0; i < cols_C; i++) {
        int found = 0;
        for (int j = 0; j < B_size; j++) {
            if (i == B[j]) {
                found = 1;
                break;
            }
        }
        if (!found) unred[unred_size++] = i;
    }

    int add[WCA_MAX_DIM];
    int add_size = 0;
    double *C_unpos = (double *)_mm_malloc(unpos_size * sizeof(double), 32);
    double *D_unpos = (double *)_mm_malloc(unpos_size * cols_D * sizeof(double), 32);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < unred_size; i++) {
        int k = unred[i];
        for (int r = 0; r < unpos_size; r++) {
            C_unpos[r] = C[unpos[r] * cols_C + k];
            for (int j = 0; j < cols_D; j++) {
                D_unpos[r * cols_D + j] = D[unpos[r] * cols_D + j];
            }
        }
        int pos_unpos_size;
        int *pos_unpos = positive_region(C_unpos, D_unpos, unpos_size, 1, cols_D, &pos_unpos_size, cache);
        if (pos_unpos_size == unpos_size) {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            add[add_size++] = k;
        }
        _mm_free(pos_unpos);
    }
    _mm_free(C_unpos);
    _mm_free(D_unpos);

    for (int i = 0; i < add_size; i++) {
        int subspace[WCA_MAX_DIM];
        int subspace_size = B_size + 1;
        #pragma unroll
        for (int j = 0; j < B_size; j++) subspace[j] = B[j];
        subspace[B_size] = add[i];

        int U_size = rows;
        int U[WCA_MAX_DIM];
        #pragma unroll
        for (int r = 0; r < U_size; r++) U[r] = r;

        for (int j = 0; j < B_size; j++) {
            int testB[WCA_MAX_DIM];
            int testB_size = 0;
            #pragma unroll
            for (int s = 0; s < subspace_size; s++) {
                if (subspace[s] != B[j]) testB[testB_size++] = subspace[s];
            }
            double *C_U = (double *)_mm_malloc(U_size * testB_size * sizeof(double), 32);
            double *D_U = (double *)_mm_malloc(U_size * cols_D * sizeof(double), 32);
            #pragma omp parallel for
            for (int r = 0; r < U_size; r++) {
                for (int s = 0; s < testB_size; s++) {
                    C_U[r * testB_size + s] = C[U[r] * cols_C + testB[s]];
                }
                for (int s = 0; s < cols_D; s++) {
                    D_U[r * cols_D + s] = D[U[r] * cols_D + s];
                }
            }
            int pos_U_size;
            int *pos_U = positive_region(C_U, D_U, U_size, testB_size, cols_D, &pos_U_size, cache);
            if (pos_U_size == U_size) {
                *v = add[i];
                *u = B[j];
                _mm_free(C_U);
                _mm_free(D_U);
                _mm_free(pos_U);
                break;
            }
            _mm_free(C_U);
            _mm_free(D_U);
            _mm_free(pos_U);
        }
        if (*u != 0 || *v != 0) break;
    }

    _mm_free(pos);
    _mm_free(unpos);
}

// LSAR-ASP with aggressive early stopping
int* lsar_asp(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    int *red = fast_red(C, D, rows, cols_C, cols_D, size, cache);
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size, cache);
    int remove_set[WCA_MAX_DIM];
    int remove_set_size = 0;
    int t = 0;

    while (remove_set_size < *size && t < LSAR_MAX_ITER) {
        int diff_red[WCA_MAX_DIM];
        int diff_red_size = 0;
        #pragma unroll
        for (int i = 0; i < *size; i++) {
            int found = 0;
            for (int j = 0; j < remove_set_size; j++) {
                if (red[i] == remove_set[j]) {
                    found = 1;
                    break;
                }
            }
            if (!found) diff_red[diff_red_size++] = red[i];
        }
        if (diff_red_size == 0) break;

        int a = diff_red[rand() % diff_red_size];
        int diff_red_a[WCA_MAX_DIM];
        int diff_red_a_size = 0;
        #pragma unroll
        for (int i = 0; i < *size; i++) {
            if (red[i] != a) diff_red_a[diff_red_a_size++] = red[i];
        }

        double *C_diff_red_a = (double *)_mm_malloc(rows * diff_red_a_size * sizeof(double), 32);
        #pragma omp parallel for
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < diff_red_a_size; j++) {
                C_diff_red_a[r * diff_red_a_size + j] = C[r * cols_C + diff_red_a[j]];
            }
        }
        int pos_diff_size;
        int *pos_diff = positive_region(C_diff_red_a, D, rows, diff_red_a_size, cols_D, &pos_diff_size, cache);
        if (pos_diff_size == pos_size) {
            _mm_free(red);
            red = (int *)_mm_malloc(diff_red_a_size * sizeof(int), 32);
            memcpy(red, diff_red_a, diff_red_a_size * sizeof(int));
            *size = diff_red_a_size;
        } else {
            int u, v;
            aps_mechanism(C, D, rows, cols_C, cols_D, diff_red_a, diff_red_a_size, &u, &v, cache);
            if (u == 0 && v == 0) {
                remove_set[remove_set_size++] = a;
            } else {
                int new_red[WCA_MAX_DIM];
                int new_red_size = 0;
                #pragma unroll
                for (int i = 0; i < *size; i++) {
                    if (red[i] != a && red[i] != u) new_red[new_red_size++] = red[i];
                }
                new_red[new_red_size++] = v;
                _mm_free(red);
                red = (int *)_mm_malloc(new_red_size * sizeof(int), 32);
                memcpy(red, new_red, new_red_size * sizeof(int));
                *size = new_red_size;
                remove_set_size = 0;
            }
        }
        _mm_free(C_diff_red_a);
        _mm_free(pos_diff);
        t++;
        if (*size <= 1 || pos_diff_size >= pos_size - CONVERGENCE_THRESHOLD) break;
    }

    _mm_free(pos);
    return red;
}

// Main WCA optimization with pre-allocated buffers
void WCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));
    initialize_streams(opt);

    // Pre-allocate all buffers
    double *C = (double *)_mm_malloc(opt->population_size * opt->dim * sizeof(double), 32);
    if (!C) exit(1);
    double *D = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    if (!D) exit(1);
    int *cache = (int *)_mm_malloc(2 * opt->population_size * sizeof(int), 32);
    if (!cache) exit(1);
    double *C_red = (double *)_mm_malloc(opt->population_size * opt->dim * sizeof(double), 32);
    if (!C_red) exit(1);
    double *C_unpos = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    if (!C_unpos) exit(1);
    double *D_unpos = (double *)_mm_malloc(opt->population_size * sizeof(double), 32);
    if (!D_unpos) exit(1);
    int *pos_temp = (int *)_mm_malloc(opt->population_size * sizeof(int), 32);
    if (!pos_temp) exit(1);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_streams(opt, objective_function);

        // Populate C and D
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            for (int j = 0; j < opt->dim; j++) {
                C[i * opt->dim + j] = opt->population[i].position[j];
            }
            D[i] = opt->population[i].fitness;
        }

        // LSAR-ASP
        int red_size;
        int *red = lsar_asp(C, D, opt->population_size, opt->dim, 1, &red_size, cache);

        // Update streams with SIMD
        #ifdef __AVX2__
        __m256d best_vec, pos_vec, bound_vec, rand_vec;
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double_wca(0.0, 1.0) < 0.5) {
                for (int j = 0; j < red_size; j++) {
                    int idx = red[j];
                    double r = PERTURBATION_FACTOR * rand_double_wca(-1.0, 1.0) * (opt->bounds[2 * idx + 1] - opt->bounds[2 * idx]);
                    opt->population[i].position[idx] = opt->best_solution.position[idx] + r;
                }
            }
        }
        #else
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double_wca(0.0, 1.0) < 0.5) {
                #pragma omp simd
                for (int j = 0; j < red_size; j++) {
                    int idx = red[j];
                    opt->population[i].position[idx] = opt->best_solution.position[idx] +
                        PERTURBATION_FACTOR * rand_double_wca(-1.0, 1.0) * (opt->bounds[2 * idx + 1] - opt->bounds[2 * idx]);
                }
            }
        }
        #endif

        wca_enforce_bound_constraints(opt);
        _mm_free(red);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    _mm_free(C);
    _mm_free(D);
    _mm_free(cache);
    _mm_free(C_red);
    _mm_free(C_unpos);
    _mm_free(D_unpos);
    _mm_free(pos_temp);
}
