#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

// Static buffers for memory reuse
static double* buffer_A = NULL;
static double* buffer_B = NULL;
static double* buffer_A_centered = NULL;
static double* buffer_B_centered = NULL;
static double* buffer_row_mean = NULL;
static double* buffer_col_mean = NULL;
static double* buffer_dist_x = NULL;
static double* buffer_dist_y = NULL;
static double* buffer_K = NULL;
static double* buffer_L = NULL;
static double* buffer_H = NULL;
static double* buffer_HK = NULL;
static double* buffer_HL = NULL;
static double* buffer_temp = NULL;
static double* buffer_x_ranks = NULL;
static double* buffer_y_ranks = NULL;
static double* buffer_x_norm = NULL;
static double* buffer_y_norm = NULL;
static double* buffer_diff_xx = NULL;
static double* buffer_diff_yy = NULL;
static size_t buffer_size = 0;

void resize_buffer(double** buffer, size_t size) {
    if (*buffer == NULL || buffer_size < size) {
        free(*buffer);
        *buffer = (double*)malloc(size * sizeof(double));
        buffer_size = size > buffer_size ? size : buffer_size;
    }
}

double compute_distance_correlation(double* x, double* y, int n) {
    if (n < 2) return 0.0;

    size_t n_square = n * n;
    size_t n_size = n;

    resize_buffer(&buffer_A, n_square);
    resize_buffer(&buffer_B, n_square);
    resize_buffer(&buffer_A_centered, n_square);
    resize_buffer(&buffer_B_centered, n_square);
    resize_buffer(&buffer_row_mean, n_size);
    resize_buffer(&buffer_col_mean, n_size);

    if (!buffer_A || !buffer_B || !buffer_A_centered || !buffer_B_centered || 
        !buffer_row_mean || !buffer_col_mean) {
        return 0.0;
    }

    double A_mean = 0.0;
    #pragma omp parallel for reduction(+:A_mean)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_A[i * n + j] = fabs(x[i] - x[j]);
            A_mean += buffer_A[i * n + j];
        }
    }
    A_mean /= (n * n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        double col_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += buffer_A[i * n + j];
            col_sum += buffer_A[j * n + i];
        }
        buffer_row_mean[i] = row_sum / n;
        buffer_col_mean[i] = col_sum / n;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_A_centered[i * n + j] = buffer_A[i * n + j] - buffer_row_mean[i] - buffer_col_mean[j] + A_mean;
        }
    }

    double B_mean = 0.0;
    #pragma omp parallel for reduction(+:B_mean)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_B[i * n + j] = fabs(y[i] - y[j]);
            B_mean += buffer_B[i * n + j];
        }
    }
    B_mean /= (n * n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        double col_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += buffer_B[i * n + j];
            col_sum += buffer_B[j * n + i];
        }
        buffer_row_mean[i] = row_sum / n;
        buffer_col_mean[i] = col_sum / n;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_B_centered[i * n + j] = buffer_B[i * n + j] - buffer_row_mean[i] - buffer_col_mean[j] + B_mean;
        }
    }

    double dCov = 0.0;
    #pragma omp parallel for reduction(+:dCov)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dCov += buffer_A_centered[i * n + j] * buffer_B_centered[i * n + j];
        }
    }
    dCov = sqrt(fabs(dCov / (n * n)));

    double dVar_x = 0.0;
    #pragma omp parallel for reduction(+:dVar_x)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dVar_x += buffer_A_centered[i * n + j] * buffer_A_centered[i * n + j];
        }
    }
    dVar_x = sqrt(fabs(dVar_x / (n * n)));

    double dVar_y = 0.0;
    #pragma omp parallel for reduction(+:dVar_y)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dVar_y += buffer_B_centered[i * n + j] * buffer_B_centered[i * n + j];
        }
    }
    dVar_y = sqrt(fabs(dVar_y / (n * n)));

    double dCor = 0.0;
    if (dVar_x > 0 && dVar_y > 0) {
        dCor = dCov / sqrt(dVar_x * dVar_y);
    }

    return dCor;
}

void compute_quantiles(double* data, int n, int q, double* quantiles) {
    resize_buffer(&buffer_temp, n);
    memcpy(buffer_temp, data, n * sizeof(double));
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (buffer_temp[j] > buffer_temp[j + 1]) {
                double temp = buffer_temp[j];
                buffer_temp[j] = buffer_temp[j + 1];
                buffer_temp[j + 1] = temp;
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < q; i++) {
        double idx = i * (n - 1.0) / (q - 1);
        int lower = (int)idx;
        double frac = idx - lower;
        if (lower + 1 < n) {
            quantiles[i] = buffer_temp[lower] * (1 - frac) + buffer_temp[lower + 1] * frac;
        } else {
            quantiles[i] = buffer_temp[lower];
        }
    }
}

double compute_mic(double* x, double* y, int n, int max_bins) {
    if (n < 2) return 0.0;

    max_bins = max_bins < (int)(ceil(pow(n, 0.6))) ? max_bins : (int)(ceil(pow(n, 0.6)));
    double mic = 0.0;

    #pragma omp parallel
    {
        double local_mic = 0.0;
        #pragma omp for schedule(dynamic)
        for (int i = 2; i <= max_bins; i++) {
            for (int j = 2; j <= max_bins; j++) {
                if (i * j > max_bins * max_bins) continue;

                double* x_quantiles = (double*)malloc(i * sizeof(double));
                double* y_quantiles = (double*)malloc(j * sizeof(double));
                compute_quantiles(x, n, i, x_quantiles);
                compute_quantiles(y, n, j, y_quantiles);

                int* x_bins = (int*)calloc(n, sizeof(int));
                int* y_bins = (int*)calloc(n, sizeof(int));
                #pragma omp parallel for
                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < i - 1; l++) {
                        if (x[k] <= x_quantiles[l + 1]) {
                            x_bins[k] = l;
                            break;
                        }
                    }
                    if (x[k] > x_quantiles[i - 1]) x_bins[k] = i - 1;
                    for (int l = 0; l < j - 1; l++) {
                        if (y[k] <= y_quantiles[l + 1]) {
                            y_bins[k] = l;
                            break;
                        }
                    }
                    if (y[k] > y_quantiles[j - 1]) y_bins[k] = j - 1;
                }

                double* joint_hist = (double*)calloc(i * j, sizeof(double));
                #pragma omp parallel for
                for (int k = 0; k < n; k++) {
                    #pragma omp atomic
                    joint_hist[x_bins[k] * j + y_bins[k]] += 1.0 / n;
                }

                double* x_hist = (double*)calloc(i, sizeof(double));
                double* y_hist = (double*)calloc(j, sizeof(double));
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < j; l++) {
                        x_hist[k] += joint_hist[k * j + l];
                        y_hist[l] += joint_hist[k * j + l];
                    }
                }

                double mi = 0.0;
                #pragma omp parallel for reduction(+:mi)
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < j; l++) {
                        if (joint_hist[k * j + l] > 0) {
                            mi += joint_hist[k * j + l] * log2(joint_hist[k * j + l] / (x_hist[k] * y_hist[l] + 1e-10));
                        }
                    }
                }

                double norm_mi = mi / log2((double)(i < j ? i : j));
                if (norm_mi > local_mic) local_mic = norm_mi;

                free(x_quantiles); free(y_quantiles);
                free(x_bins); free(y_bins);
                free(joint_hist); free(x_hist); free(y_hist);
            }
        }
        #pragma omp critical
        if (local_mic > mic) mic = local_mic;
    }

    return mic;
}

double compute_hsic(double* x, double* y, int n) {
    if (n < 2) return 0.0;

    size_t n_square = n * n;
    size_t n_minus_one = n * (n - 1);

    resize_buffer(&buffer_dist_x, n_square);
    resize_buffer(&buffer_dist_y, n_square);
    resize_buffer(&buffer_K, n_square);
    resize_buffer(&buffer_L, n_square);
    resize_buffer(&buffer_H, n_square);
    resize_buffer(&buffer_HK, n_square);
    resize_buffer(&buffer_HL, n_square);
    resize_buffer(&buffer_temp, n_minus_one);

    if (!buffer_dist_x || !buffer_dist_y || !buffer_K || !buffer_L || 
        !buffer_H || !buffer_HK || !buffer_HL || !buffer_temp) {
        return 0.0;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_dist_x[i * n + j] = fabs(x[i] - x[j]);
            buffer_dist_y[i * n + j] = fabs(y[i] - y[j]);
        }
    }

    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (buffer_dist_x[i * n + j] > 0) buffer_temp[idx++] = buffer_dist_x[i * n + j];
        }
    }
    double sigma_x = 0.0;
    if (idx > 0) {
        for (int i = 0; i < idx - 1; i++) {
            for (int j = 0; j < idx - i - 1; j++) {
                if (buffer_temp[j] > buffer_temp[j + 1]) {
                    double t = buffer_temp[j];
                    buffer_temp[j] = buffer_temp[j + 1];
                    buffer_temp[j + 1] = t;
                }
            }
        }
        sigma_x = buffer_temp[idx / 2];
    } else {
        sigma_x = 1.0;
    }

    idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (buffer_dist_y[i * n + j] > 0) buffer_temp[idx++] = buffer_dist_y[i * n + j];
        }
    }
    double sigma_y = 0.0;
    if (idx > 0) {
        for (int i = 0; i < idx - 1; i++) {
            for (int j = 0; j < idx - i - 1; j++) {
                if (buffer_temp[j] > buffer_temp[j + 1]) {
                    double t = buffer_temp[j];
                    buffer_temp[j] = buffer_temp[j + 1];
                    buffer_temp[j + 1] = t;
                }
            }
        }
        sigma_y = buffer_temp[idx / 2];
    } else {
        sigma_y = 1.0;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_K[i * n + j] = exp(-buffer_dist_x[i * n + j] * buffer_dist_x[i * n + j] / (2 * sigma_x * sigma_x + 1e-10));
            buffer_L[i * n + j] = exp(-buffer_dist_y[i * n + j] * buffer_dist_y[i * n + j] / (2 * sigma_y * sigma_y + 1e-10));
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_H[i * n + j] = (i == j) ? 1.0 - 1.0 / n : -1.0 / n;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += buffer_H[i * n + k] * buffer_K[k * n + j];
            }
            buffer_HK[i * n + j] = sum;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += buffer_H[i * n + k] * buffer_L[k * n + j];
            }
            buffer_HL[i * n + j] = sum;
        }
    }

    double hsic = 0.0;
    #pragma omp parallel for reduction(+:hsic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hsic += buffer_HK[i * n + j] * buffer_HL[j * n + i];
        }
    }
    hsic /= (n * n);

    return sqrt(hsic > 0 ? hsic : 0);
}

void rankdata(double* data, double* ranks, int n) {
    typedef struct { double value; int index; } pair_t;
    pair_t* pairs = (pair_t*)malloc(n * sizeof(pair_t));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        pairs[i].value = data[i];
        pairs[i].index = i;
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (pairs[j].value > pairs[j + 1].value) {
                pair_t temp = pairs[j];
                pairs[j] = pairs[j + 1];
                pairs[j + 1] = temp;
            }
        }
    }

    int i = 0;
    while (i < n) {
        int start = i;
        double sum_ranks = 0;
        int count = 0;
        while (i < n && pairs[i].value == pairs[start].value) {
            sum_ranks += i + 1;
            count++;
            i++;
        }
        double avg_rank = sum_ranks / count;
        #pragma omp parallel for
        for (int j = start; j < i; j++) {
            ranks[pairs[j].index] = avg_rank;
        }
    }
    free(pairs);
}

double compute_copula_measure(double* x, double* y, int n) {
    size_t n_size = n;

    resize_buffer(&buffer_x_ranks, n_size);
    resize_buffer(&buffer_y_ranks, n_size);
    resize_buffer(&buffer_x_norm, n_size);
    resize_buffer(&buffer_y_norm, n_size);

    if (!buffer_x_ranks || !buffer_y_ranks || !buffer_x_norm || !buffer_y_norm) {
        return 0.0;
    }

    rankdata(x, buffer_x_ranks, n);
    rankdata(y, buffer_y_ranks, n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        buffer_x_norm[i] = buffer_x_ranks[i] / (n + 1);
        buffer_y_norm[i] = buffer_y_ranks[i] / (n + 1);
    }

    double mean_x = 0.0, mean_y = 0.0;
    #pragma omp parallel for reduction(+:mean_x,mean_y)
    for (int i = 0; i < n; i++) {
        mean_x += buffer_x_norm[i];
        mean_y += buffer_y_norm[i];
    }
    mean_x /= n;
    mean_y /= n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    #pragma omp parallel for reduction(+:cov,var_x,var_y)
    for (int i = 0; i < n; i++) {
        cov += (buffer_x_norm[i] - mean_x) * (buffer_y_norm[i] - mean_y);
        var_x += (buffer_x_norm[i] - mean_x) * (buffer_x_norm[i] - mean_x);
        var_y += (buffer_y_norm[i] - mean_y) * (buffer_y_norm[i] - mean_y);
    }
    cov /= n;
    var_x /= n;
    var_y /= n;

    double corr = 0.0;
    if (var_x > 0 && var_y > 0) {
        corr = cov / sqrt(var_x * var_y);
    }

    return fabs(corr);
}

double compute_energy_distance_correlation(double* x, double* y, int n) {
    if (n < 2) return 0.0;

    size_t n_square = n * n;

    resize_buffer(&buffer_diff_xx, n_square);
    resize_buffer(&buffer_diff_yy, n_square);

    if (!buffer_diff_xx || !buffer_diff_yy) {
        return 0.0;
    }

    double dist_xy = 0.0, dist_xx = 0.0, dist_yy = 0.0;
    #pragma omp parallel for reduction(+:dist_xy,dist_xx,dist_yy)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist_xy += fabs(x[i] - y[j]);
            dist_xx += fabs(x[i] - x[j]);
            dist_yy += fabs(y[i] - y[j]);
        }
    }
    dist_xy /= (n * n);
    dist_xx /= (n * n);
    dist_yy /= (n * n);

    double energy_dist = 2 * dist_xy - dist_xx - dist_yy;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer_diff_xx[i * n + j] = fabs(x[i] - x[j]) - dist_xx;
            buffer_diff_yy[i * n + j] = fabs(y[i] - y[j]) - dist_yy;
        }
    }

    double var_xx = 0.0, var_yy = 0.0;
    #pragma omp parallel for reduction(+:var_xx)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            var_xx += buffer_diff_xx[i * n + j] * buffer_diff_xx[i * n + j];
        }
    }
    var_xx = sqrt(var_xx / (n * n));

    #pragma omp parallel for reduction(+:var_yy)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            var_yy += buffer_diff_yy[i * n + j] * buffer_diff_yy[i * n + j];
        }
    }
    var_yy = sqrt(var_yy / (n * n));

    double energy_corr = 0.0;
    if (var_xx > 0 && var_yy > 0) {
        energy_corr = energy_dist / sqrt(var_xx * var_yy);
    }

    return fabs(energy_corr);
}

// Cleanup function to free buffers (call at program exit if needed)
void cleanup_buffers() {
    free(buffer_A); free(buffer_B); free(buffer_A_centered); free(buffer_B_centered);
    free(buffer_row_mean); free(buffer_col_mean); free(buffer_dist_x); free(buffer_dist_y);
    free(buffer_K); free(buffer_L); free(buffer_H); free(buffer_HK); free(buffer_HL);
    free(buffer_temp); free(buffer_x_ranks); free(buffer_y_ranks); 
    free(buffer_x_norm); free(buffer_y_norm); free(buffer_diff_xx); free(buffer_diff_yy);
    buffer_A = buffer_B = buffer_A_centered = buffer_B_centered = NULL;
    buffer_row_mean = buffer_col_mean = buffer_dist_x = buffer_dist_y = NULL;
    buffer_K = buffer_L = buffer_H = buffer_HK = buffer_HL = buffer_temp = NULL;
    buffer_x_ranks = buffer_y_ranks = buffer_x_norm = buffer_y_norm = NULL;
    buffer_diff_xx = buffer_diff_yy = NULL;
    buffer_size = 0;
}
