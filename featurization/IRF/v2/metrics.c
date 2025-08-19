#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

double compute_distance_correlation(double* x, double* y, int n) {
    if (n < 2) return 0.0;

    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* A_centered = (double*)malloc(n * n * sizeof(double));
    double* B_centered = (double*)malloc(n * n * sizeof(double));

    if (!A || !B || !A_centered || !B_centered) {
        free(A); free(B); free(A_centered); free(B_centered);
        return 0.0;
    }

    double A_mean = 0.0;
    #pragma omp parallel for reduction(+:A_mean)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = fabs(x[i] - x[j]);
            A_mean += A[i * n + j];
        }
    }
    A_mean /= (n * n);

    double* A_row_mean = (double*)calloc(n, sizeof(double));
    double* A_col_mean = (double*)calloc(n, sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        double col_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += A[i * n + j];
            col_sum += A[j * n + i];
        }
        A_row_mean[i] = row_sum / n;
        A_col_mean[i] = col_sum / n;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_centered[i * n + j] = A[i * n + j] - A_row_mean[i] - A_col_mean[j] + A_mean;
        }
    }

    double B_mean = 0.0;
    #pragma omp parallel for reduction(+:B_mean)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = fabs(y[i] - y[j]);
            B_mean += B[i * n + j];
        }
    }
    B_mean /= (n * n);

    double* B_row_mean = (double*)calloc(n, sizeof(double));
    double* B_col_mean = (double*)calloc(n, sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        double col_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += B[i * n + j];
            col_sum += B[j * n + i];
        }
        B_row_mean[i] = row_sum / n;
        B_col_mean[i] = col_sum / n;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B_centered[i * n + j] = B[i * n + j] - B_row_mean[i] - B_col_mean[j] + B_mean;
        }
    }

    double dCov = 0.0;
    #pragma omp parallel for reduction(+:dCov)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dCov += A_centered[i * n + j] * B_centered[i * n + j];
        }
    }
    dCov = sqrt(fabs(dCov / (n * n)));

    double dVar_x = 0.0;
    #pragma omp parallel for reduction(+:dVar_x)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dVar_x += A_centered[i * n + j] * A_centered[i * n + j];
        }
    }
    dVar_x = sqrt(fabs(dVar_x / (n * n)));

    double dVar_y = 0.0;
    #pragma omp parallel for reduction(+:dVar_y)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dVar_y += B_centered[i * n + j] * B_centered[i * n + j];
        }
    }
    dVar_y = sqrt(fabs(dVar_y / (n * n)));

    double dCor = 0.0;
    if (dVar_x > 0 && dVar_y > 0) {
        dCor = dCov / sqrt(dVar_x * dVar_y);
    }

    free(A); free(B); free(A_centered); free(B_centered);
    free(A_row_mean); free(A_col_mean);
    free(B_row_mean); free(B_col_mean);

    return dCor;
}

void compute_quantiles(double* data, int n, int q, double* quantiles) {
    double* sorted = (double*)malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                double temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < q; i++) {
        double idx = i * (n - 1.0) / (q - 1);
        int lower = (int)idx;
        double frac = idx - lower;
        if (lower + 1 < n) {
            quantiles[i] = sorted[lower] * (1 - frac) + sorted[lower + 1] * frac;
        } else {
            quantiles[i] = sorted[lower];
        }
    }
    free(sorted);
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

    double* dist_x = (double*)malloc(n * n * sizeof(double));
    double* dist_y = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist_x[i * n + j] = fabs(x[i] - x[j]);
            dist_y[i * n + j] = fabs(y[i] - y[j]);
        }
    }

    double* temp = (double*)malloc(n * (n - 1) * sizeof(double));
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist_x[i * n + j] > 0) temp[idx++] = dist_x[i * n + j];
        }
    }
    double sigma_x = 0.0;
    if (idx > 0) {
        for (int i = 0; i < idx - 1; i++) {
            for (int j = 0; j < idx - i - 1; j++) {
                if (temp[j] > temp[j + 1]) {
                    double t = temp[j];
                    temp[j] = temp[j + 1];
                    temp[j + 1] = t;
                }
            }
        }
        sigma_x = temp[idx / 2];
    } else {
        sigma_x = 1.0;
    }

    idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist_y[i * n + j] > 0) temp[idx++] = dist_y[i * n + j];
        }
    }
    double sigma_y = 0.0;
    if (idx > 0) {
        for (int i = 0; i < idx - 1; i++) {
            for (int j = 0; j < idx - i - 1; j++) {
                if (temp[j] > temp[j + 1]) {
                    double t = temp[j];
                    temp[j] = temp[j + 1];
                    temp[j + 1] = t;
                }
            }
        }
        sigma_y = temp[idx / 2];
    } else {
        sigma_y = 1.0;
    }
    free(temp);

    double* K = (double*)malloc(n * n * sizeof(double));
    double* L = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            K[i * n + j] = exp(-dist_x[i * n + j] * dist_x[i * n + j] / (2 * sigma_x * sigma_x + 1e-10));
            L[i * n + j] = exp(-dist_y[i * n + j] * dist_y[i * n + j] / (2 * sigma_y * sigma_y + 1e-10));
        }
    }

    double* H = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i * n + j] = (i == j) ? 1.0 - 1.0 / n : -1.0 / n;
        }
    }

    double* HK = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += H[i * n + k] * K[k * n + j];
            }
            HK[i * n + j] = sum;
        }
    }

    double* HL = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += H[i * n + k] * L[k * n + j];
            }
            HL[i * n + j] = sum;
        }
    }

    double hsic = 0.0;
    #pragma omp parallel for reduction(+:hsic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hsic += HK[i * n + j] * HL[j * n + i];
        }
    }
    hsic /= (n * n);

    free(dist_x); free(dist_y);
    free(K); free(L); free(H); free(HK); free(HL);

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
    double* x_ranks = (double*)malloc(n * sizeof(double));
    double* y_ranks = (double*)malloc(n * sizeof(double));
    rankdata(x, x_ranks, n);
    rankdata(y, y_ranks, n);

    double* x_norm = (double*)malloc(n * sizeof(double));
    double* y_norm = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x_norm[i] = x_ranks[i] / (n + 1);
        y_norm[i] = y_ranks[i] / (n + 1);
    }

    double mean_x = 0.0, mean_y = 0.0;
    #pragma omp parallel for reduction(+:mean_x,mean_y)
    for (int i = 0; i < n; i++) {
        mean_x += x_norm[i];
        mean_y += y_norm[i];
    }
    mean_x /= n;
    mean_y /= n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    #pragma omp parallel for reduction(+:cov,var_x,var_y)
    for (int i = 0; i < n; i++) {
        cov += (x_norm[i] - mean_x) * (y_norm[i] - mean_y);
        var_x += (x_norm[i] - mean_x) * (x_norm[i] - mean_x);
        var_y += (y_norm[i] - mean_y) * (y_norm[i] - mean_y);
    }
    cov /= n;
    var_x /= n;
    var_y /= n;

    double corr = 0.0;
    if (var_x > 0 && var_y > 0) {
        corr = cov / sqrt(var_x * var_y);
    }

    free(x_ranks); free(y_ranks);
    free(x_norm); free(y_norm);

    return fabs(corr);
}

double compute_energy_distance_correlation(double* x, double* y, int n) {
    if (n < 2) return 0.0;

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

    double* diff_xx = (double*)malloc(n * n * sizeof(double));
    double* diff_yy = (double*)malloc(n * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            diff_xx[i * n + j] = fabs(x[i] - x[j]) - dist_xx;
            diff_yy[i * n + j] = fabs(y[i] - y[j]) - dist_yy;
        }
    }

    double var_xx = 0.0, var_yy = 0.0;
    #pragma omp parallel for reduction(+:var_xx)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            var_xx += diff_xx[i * n + j] * diff_xx[i * n + j];
        }
    }
    var_xx = sqrt(var_xx / (n * n));

    #pragma omp parallel for reduction(+:var_yy)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            var_yy += diff_yy[i * n + j] * diff_yy[i * n + j];
        }
    }
    var_yy = sqrt(var_yy / (n * n));

    double energy_corr = 0.0;
    if (var_xx > 0 && var_yy > 0) {
        energy_corr = energy_dist / sqrt(var_xx * var_yy);
    }

    free(diff_xx); free(diff_yy);

    return fabs(energy_corr);
}
