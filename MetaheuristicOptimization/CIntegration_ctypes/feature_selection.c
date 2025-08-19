#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void compute_variance(double *data, int rows, int cols, double *variances) {
    // Compute variance for each column
    for (int j = 0; j < cols; j++) {
        double mean = 0.0;
        for (int i = 0; i < rows; i++) {
            mean += data[i * cols + j];
        }
        mean /= rows;
        
        double var = 0.0;
        for (int i = 0; i < rows; i++) {
            double diff = data[i * cols + j] - mean;
            var += diff * diff;
        }
        variances[j] = var / (rows - 1);
    }
}

void select_top_features(double *scores, int n, int top_n, int *selected_indices) {
    // Simple selection sort to get top_n indices
    double *temp_scores = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        temp_scores[i] = scores[i];
    }
    
    for (int i = 0; i < top_n && i < n; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (temp_scores[j] > temp_scores[max_idx]) {
                max_idx = j;
            }
        }
        selected_indices[i] = max_idx;
        temp_scores[max_idx] = -1.0; // Mark as selected
    }
    free(temp_scores);
}

void compute_correlation(double *data, int rows, int cols, double *corr_matrix) {
    // Compute Pearson correlation matrix
    double *means = (double *)calloc(cols, sizeof(double));
    double *stds = (double *)calloc(cols, sizeof(double));
    
    // Compute means and standard deviations
    for (int j = 0; j < cols; j++) {
        double mean = 0.0;
        for (int i = 0; i < rows; i++) {
            mean += data[i * cols + j];
        }
        mean /= rows;
        means[j] = mean;
        
        double var = 0.0;
        for (int i = 0; i < rows; i++) {
            double diff = data[i * cols + j] - mean;
            var += diff * diff;
        }
        stds[j] = sqrt(var / (rows - 1));
    }
    
    // Compute correlations
    for (int j1 = 0; j1 < cols; j1++) {
        for (int j2 = j1; j2 < cols; j2++) {
            double cov = 0.0;
            for (int i = 0; i < rows; i++) {
                cov += (data[i * cols + j1] - means[j1]) * (data[i * cols + j2] - means[j2]);
            }
            cov /= (rows - 1);
            double corr = (stds[j1] > 0 && stds[j2] > 0) ? cov / (stds[j1] * stds[j2]) : 0.0;
            corr_matrix[j1 * cols + j2] = corr;
            corr_matrix[j2 * cols + j1] = corr;
        }
    }
    
    free(means);
    free(stds);
}

void compute_chi2(double *X, int rows, int cols, int *y, int n_classes, double *scores) {
    // Compute Chi-Square scores for each feature
    for (int j = 0; j < cols; j++) {
        double chi2 = 0.0;
        // For each class
        for (int c = 0; c < n_classes; c++) {
            double observed_pos = 0.0, observed_neg = 0.0;
            double expected_pos = 0.0, expected_neg = 0.0;
            int class_count = 0;
            for (int i = 0; i < rows; i++) {
                if (y[i] == c) {
                    class_count++;
                    if (X[i * cols + j] > 0) observed_pos++;
                    else observed_neg++;
                }
            }
            double total_pos = 0.0, total_neg = 0.0;
            for (int i = 0; i < rows; i++) {
                if (X[i * cols + j] > 0) total_pos++;
                else total_neg++;
            }
            if (class_count > 0) {
                expected_pos = (total_pos * class_count) / rows;
                expected_neg = (total_neg * class_count) / rows;
                if (expected_pos > 0) {
                    chi2 += (observed_pos - expected_pos) * (observed_pos - expected_pos) / expected_pos;
                }
                if (expected_neg > 0) {
                    chi2 += (observed_neg - expected_neg) * (observed_neg - expected_neg) / expected_neg;
                }
            }
        }
        scores[j] = chi2;
    }
}
