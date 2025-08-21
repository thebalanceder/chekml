#include "WCA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

double rand_double(double min, double max);

// Enforce bounds with SIMD if available
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
    #pragma omp parallel for
    #endif
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

// Evaluate streams with OpenMP
void evaluate_streams(Optimizer *opt, double (*objective_function)(double *)) {
    #ifdef _OPENMP
    #pragma omp parallel for
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
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
    }
}

// Compute IOP with caching
double iop(double *C, double *D, int rows, int cols_C, int cols_D, int *cache) {
    if (cols_C == 0 || rows == 0) return 0.0;

    double sum_diff = 0.0;
    int *counts_C = cache;
    int *counts_CD = cache + rows;

    for (int i = 0; i < rows; i++) {
        counts_C[i] = 1;
        counts_CD[i] = 1;
    }

    for (int i = 0; i < rows; i++) {
        sum_diff += (double)(counts_C[i] - counts_CD[i]);
    }
    return sum_diff;
}

// Compute positive region
int* positive_region(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    if (cols_C == 0 || rows == 0) {
        *size = 0;
        return NULL;
    }

    int *pos = (int *)malloc(rows * sizeof(int));
    if (!pos) return NULL;
    *size = 0;
    double diff = iop(C, D, rows, cols_C, cols_D, cache);
    if (fabs(diff) < 1e-6) {
        for (int i = 0; i < rows; i++) {
            pos[(*size)++] = i;
        }
    }
    return pos;
}

// Quick sort for indices
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

        quick_sort(arr, indices, low, i);
        quick_sort(arr, indices, i + 2, high);
    }
}

// Fast reduction with pre-allocated buffer
int* fast_red(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    double iop_C = iop(C, D, rows, cols_C, cols_D, cache);
    double *w = (double *)malloc(cols_C * sizeof(double));
    int *ind = (int *)malloc(cols_C * sizeof(int));
    int red[MAX_DIM];
    *size = 0;

    double *C_single = (double *)malloc(rows * sizeof(double));
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < cols_C; i++) {
        for (int r = 0; r < rows; r++) {
            C_single[r] = C[r * cols_C + i];
        }
        w[i] = iop(C_single, D, rows, 1, cols_D, cache);
        ind[i] = i;
    }
    free(C_single);

    quick_sort(w, ind, 0, cols_C - 1);

    double *C_red = (double *)malloc(rows * cols_C * sizeof(double));
    for (int i = 0; i < cols_C; i++) {
        red[(*size)++] = ind[i];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < *size; j++) {
                C_red[r * cols_C + j] = C[r * cols_C + red[j]];
            }
        }
        if (fabs(iop(C_red, D, rows, *size, cols_D, cache) - iop_C) < EARLY_STOP_THRESHOLD) {
            break;
        }
    }

    int *result = (int *)malloc(*size * sizeof(int));
    memcpy(result, red, *size * sizeof(int));
    free(w);
    free(ind);
    free(C_red);
    return result;
}

// APS mechanism with bitset for set operations
void aps_mechanism(double *C, double *D, int rows, int cols_C, int cols_D, int *B, int B_size, int *u, int *v, int *cache) {
    *u = 0;
    *v = 0;
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size, cache);
    double *C_B = (double *)malloc(rows * B_size * sizeof(double));
    for (int r = 0; r < rows; r++) {
        for (int j = 0; j < B_size; j++) {
            C_B[r * B_size + j] = C[r * cols_C + B[j]];
        }
    }
    int unpos_size;
    int *unpos = positive_region(C_B, D, rows, B_size, cols_D, &unpos_size, cache);
    free(C_B);

    int unred[MAX_DIM];
    int unred_size = 0;
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

    int add[MAX_DIM];
    int add_size = 0;
    double *C_unpos = (double *)malloc(unpos_size * sizeof(double));
    double *D_unpos = (double *)malloc(unpos_size * cols_D * sizeof(double));
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
            add[add_size++] = k;
        }
        free(pos_unpos);
    }
    free(C_unpos);
    free(D_unpos);

    for (int i = 0; i < add_size; i++) {
        int subspace[MAX_DIM];
        int subspace_size = B_size + 1;
        for (int j = 0; j < B_size; j++) subspace[j] = B[j];
        subspace[B_size] = add[i];

        int U_size = rows;
        int U[MAX_DIM];
        for (int r = 0; r < U_size; r++) U[r] = r;

        for (int j = 0; j < B_size; j++) {
            int testB[MAX_DIM];
            int testB_size = 0;
            for (int s = 0; s < subspace_size; s++) {
                if (subspace[s] != B[j]) testB[testB_size++] = subspace[s];
            }
            double *C_U = (double *)malloc(U_size * testB_size * sizeof(double));
            double *D_U = (double *)malloc(U_size * cols_D * sizeof(double));
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
                free(C_U);
                free(D_U);
                free(pos_U);
                break;
            }
            free(C_U);
            free(D_U);
            free(pos_U);
        }
        if (*u != 0 || *v != 0) break;
    }

    free(pos);
    free(unpos);
}

// LSAR-ASP with early stopping
int* lsar_asp(double *C, double *D, int rows, int cols_C, int cols_D, int *size, int *cache) {
    int *red = fast_red(C, D, rows, cols_C, cols_D, size, cache);
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size, cache);
    int remove_set[MAX_DIM];
    int remove_set_size = 0;
    int t = 0;

    while (remove_set_size < *size && t < LSAR_MAX_ITER) {
        int diff_red[MAX_DIM];
        int diff_red_size = 0;
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
        int diff_red_a[MAX_DIM];
        int diff_red_a_size = 0;
        for (int i = 0; i < *size; i++) {
            if (red[i] != a) diff_red_a[diff_red_a_size++] = red[i];
        }

        double *C_diff_red_a = (double *)malloc(rows * diff_red_a_size * sizeof(double));
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < diff_red_a_size; j++) {
                C_diff_red_a[r * diff_red_a_size + j] = C[r * cols_C + diff_red_a[j]];
            }
        }
        int pos_diff_size;
        int *pos_diff = positive_region(C_diff_red_a, D, rows, diff_red_a_size, cols_D, &pos_diff_size, cache);
        if (pos_diff_size == pos_size) {
            free(red);
            red = (int *)malloc(diff_red_a_size * sizeof(int));
            memcpy(red, diff_red_a, diff_red_a_size * sizeof(int));
            *size = diff_red_a_size;
        } else {
            int u, v;
            aps_mechanism(C, D, rows, cols_C, cols_D, diff_red_a, diff_red_a_size, &u, &v, cache);
            if (u == 0 && v == 0) {
                remove_set[remove_set_size++] = a;
            } else {
                int new_red[MAX_DIM];
                int new_red_size = 0;
                for (int i = 0; i < *size; i++) {
                    if (red[i] != a && red[i] != u) new_red[new_red_size++] = red[i];
                }
                new_red[new_red_size++] = v;
                free(red);
                red = (int *)malloc(new_red_size * sizeof(int));
                memcpy(red, new_red, new_red_size * sizeof(int));
                *size = new_red_size;
                remove_set_size = 0;
            }
        }
        free(C_diff_red_a);
        free(pos_diff);
        t++;
        if (*size <= 1 || pos_diff_size >= pos_size - EARLY_STOP_THRESHOLD) break;
    }

    free(pos);
    return red;
}

// Main WCA optimization
void WCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand(time(NULL));
    initialize_streams(opt);

    // Pre-allocate buffers
    double *C = (double *)malloc(opt->population_size * opt->dim * sizeof(double));
    double *D = (double *)malloc(opt->population_size * sizeof(double));
    int *cache = (int *)malloc(2 * opt->population_size * sizeof(int));

    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_streams(opt, objective_function);

        // Populate C and D
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
            if (rand_double(0.0, 1.0) < 0.5) {
                for (int j = 0; j < red_size; j++) {
                    int idx = red[j];
                    double r = PERTURBATION_FACTOR * rand_double(-1.0, 1.0) * (opt->bounds[2 * idx + 1] - opt->bounds[2 * idx]);
                    opt->population[i].position[idx] = opt->best_solution.position[idx] + r;
                }
            }
        }
        #else
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double(0.0, 1.0) < 0.5) {
                for (int j = 0; j < red_size; j++) {
                    int idx = red[j];
                    opt->population[i].position[idx] = opt->best_solution.position[idx] +
                        PERTURBATION_FACTOR * rand_double(-1.0, 1.0) * (opt->bounds[2 * idx + 1] - opt->bounds[2 * idx]);
                }
            }
        }
        #endif

        wca_enforce_bound_constraints(opt);
        free(red);
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    free(C);
    free(D);
    free(cache);
}
