#include "inequalities.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MIN_VALUE 1e-10

// Inequality functions
double am(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double gm(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += log(fmax(x[i], MIN_VALUE));
    return exp(sum / n);
}

double hm(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += 1.0 / fmax(x[i], MIN_VALUE);
    return n / sum;
}

double qm(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrt(sum / n);
}

double pm3(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), 3.0);
    return pow(sum / n, 1.0/3.0);
}

double pm_neg1(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += 1.0 / fmax(x[i], MIN_VALUE);
    return n / sum;
}

double lehmer2(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += val * val;
        denom += val;
    }
    return num / denom;
}

double lehmer05(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += sqrt(val);
        denom += 1.0 / sqrt(val);
    }
    return num / denom;
}

double log_mean(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return (b - a) / log(b / a);
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double identric(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return pow(a, b/(b-a)) * pow(b, a/(a-b));
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += log(fmax(x[i], MIN_VALUE));
    return exp(sum / n - 1.0);
}

double heronian(double* x, int n) {
    if (n == 2) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return (a + sqrt(a * b) + b) / 3.0;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double contra_hm(double* x, int n) {
    double sum_sq = 0.0, sum = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        sum_sq += val * val;
        sum += val;
    }
    return (sum_sq / n) / (sum / n);
}

double rms(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrt(sum / n);
}

double pm4(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), 4.0);
    return pow(sum / n, 1.0/4.0);
}

double pm2(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), 2.0);
    return pow(sum / n, 1.0/2.0);
}

double pm_neg2(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), -2.0);
    return pow(n / sum, 1.0/2.0);
}

double lehmer3(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += pow(val, 3.0);
        denom += val * val;
    }
    return num / denom;
}

double lehmer_neg1(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += 1.0 / val;
        denom += 1.0 / (val * val);
    }
    return num / denom;
}

double centroidal(double* x, int n) {
    double sum = 0.0, w_sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (i + 1) * x[i];
        w_sum += (i + 1);
    }
    return sum / w_sum;
}

double seiffert(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return (a - b) / (2.0 * asin((a - b) / (a + b)));
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double neuman_sandor(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return (a - b) / (2.0 * asinh((a - b) / (a + b)));
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double log_mean_gen(double* x, int n) {
    if (n > 1) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double a = fmax(x[i], MIN_VALUE), b = fmax(x[j], MIN_VALUE);
                sum += (a == b) ? a : (b - a) / log(b / a);
                count++;
            }
        }
        return sum / count;
    }
    return x[0];
}

double stolarsky2(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return pow((pow(b, 2.0) - pow(a, 2.0)) / (2.0 * (b - a)), 1.0);
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double pm6(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), 6.0);
    return pow(sum / n, 1.0/6.0);
}

double pm_neg3(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += pow(fmax(x[i], MIN_VALUE), -3.0);
    return pow(n / sum, 1.0/3.0);
}

double lehmer4(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += pow(val, 4.0);
        denom += pow(val, 3.0);
    }
    return num / denom;
}

double lehmer_neg2(double* x, int n) {
    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        double val = fmax(x[i], MIN_VALUE);
        num += 1.0 / (val * val);
        denom += 1.0 / pow(val, 3.0);
    }
    return num / denom;
}

double exp_mean(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp(x[i]);
    return log(sum / n);
}

double quad_entropy(double* x, int n) {
    double sum_x = 0.0;
    for (int i = 0; i < n; i++) sum_x += fmax(x[i], MIN_VALUE);
    double entropy = 0.0;
    for (int i = 0; i < n; i++) {
        double p = fmax(x[i], MIN_VALUE) / sum_x;
        entropy -= p * p * log(fmax(p, MIN_VALUE));
    }
    return entropy;
}

double wgm(double* x, int n) {
    double sum_x = 0.0;
    for (int i = 0; i < n; i++) sum_x += fmax(x[i], MIN_VALUE);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double w = fmax(x[i], MIN_VALUE) / sum_x;
        sum += w * log(fmax(x[i], MIN_VALUE));
    }
    return exp(sum);
}

double hyperbolic(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return (a + b) / (2.0 * cosh((a - b) / (a + b)));
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double stolarsky3(double* x, int n) {
    if (n == 2 && x[0] != x[1]) {
        double a = fmax(x[0], MIN_VALUE), b = fmax(x[1], MIN_VALUE);
        return pow((pow(b, 3.0) - pow(a, 3.0)) / (3.0 * (b - a)), 1.0/2.0);
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / n;
}

double midrange(double* x, int n) {
    double min_x = x[0], max_x = x[0];
    for (int i = 1; i < n; i++) {
        min_x = fmin(min_x, x[i]);
        max_x = fmax(max_x, x[i]);
    }
    return (min_x + max_x) / 2.0;
}

// Array of inequalities
Inequality inequalities[] = {
    {"am", am}, {"gm", gm}, {"hm", hm}, {"qm", qm},
    {"pm3", pm3}, {"pm_neg1", pm_neg1}, {"lehmer2", lehmer2},
    {"lehmer05", lehmer05}, {"log_mean", log_mean},
    {"identric", identric}, {"heronian", heronian},
    {"contra_hm", contra_hm}, {"rms", rms},
    {"pm4", pm4}, {"pm2", pm2}, {"pm_neg2", pm_neg2},
    {"lehmer3", lehmer3}, {"lehmer_neg1", lehmer_neg1},
    {"centroidal", centroidal}, {"seiffert", seiffert},
    {"neuman_sandor", neuman_sandor}, {"log_mean_gen", log_mean_gen},
    {"stolarsky2", stolarsky2}, {"pm6", pm6}, {"pm_neg3", pm_neg3},
    {"lehmer4", lehmer4}, {"lehmer_neg2", lehmer_neg2},
    {"exp_mean", exp_mean}, {"quad_entropy", quad_entropy},
    {"wgm", wgm}, {"hyperbolic", hyperbolic},
    {"stolarsky3", stolarsky3}, {"midrange", midrange}
};
const int num_inequalities = sizeof(inequalities) / sizeof(Inequality);

// Comparison function for sorting results
int compare_results(const void* a, const void* b) {
    double val_a = ((Result*)b)->value; // Reverse order (descending)
    double val_b = ((Result*)a)->value;
    return (val_a > val_b) - (val_a < val_b);
}

// Main function to compute features
void compute_features(double* data, int rows, int* cols, int num_cols, int level, int stage, 
                     double* output, int* output_cols, char** output_names) {
    int out_idx = 0;
    double* temp = (double*)malloc(rows * sizeof(double));
    Result* results = (Result*)malloc(num_inequalities * sizeof(Result));
    
    // For each combination level
    for (int r = 1; r <= level; r++) {
        // Generate combinations
        int* comb = (int*)malloc(r * sizeof(int));
        for (int i = 0; i < r; i++) comb[i] = i;
        
        while (comb[0] < num_cols - r + 1) {
            // Compute inequalities
            int result_count = 0;
            for (int i = 0; i < num_inequalities; i++) {
                for (int j = 0; j < rows; j++) {
                    double x[10]; // Max 10 features per combination
                    for (int k = 0; k < r; k++) {
                        x[k] = data[j * num_cols + cols[comb[k]]];
                    }
                    temp[j] = inequalities[i].func(x, r);
                }
                if (!isnan(temp[0])) { // Check if valid
                    double avg = 0.0;
                    for (int j = 0; j < rows; j++) avg += fabs(temp[j]);
                    avg /= rows;
                    results[result_count].name = inequalities[i].name;
                    results[result_count].value = avg;
                    result_count++;
                }
            }
            
            // Sort and select top stage inequalities
            qsort(results, result_count, sizeof(Result), compare_results);
            int top_count = (stage < result_count) ? stage : result_count;
            
            // Copy top features to output
            for (int i = 0; i < top_count; i++) {
                for (int j = 0; j < num_inequalities; j++) {
                    if (strcmp(results[i].name, inequalities[j].name) == 0) {
                        for (int k = 0; k < rows; k++) {
                            double x[10];
                            for (int m = 0; m < r; m++) {
                                x[m] = data[k * num_cols + cols[comb[m]]];
                            }
                            output[out_idx * rows + k] = inequalities[j].func(x, r);
                        }
                        // Construct feature name
                        char name[256] = "";
                        for (int m = 0; m < r; m++) {
                            strcat(name, "_");
                            char idx[10];
                            sprintf(idx, "%d", cols[comb[m]]);
                            strcat(name, idx);
                        }
                        strcat(name, "_");
                        strcat(name, results[i].name);
                        output_names[out_idx] = strdup(name + 1); // Skip leading underscore
                        out_idx++;
                        break;
                    }
                }
            }
            
            // Next combination
            int k = r - 1;
            while (k >= 0 && comb[k] >= num_cols - r + k) k--;
            if (k < 0) break;
            comb[k]++;
            for (int i = k + 1; i < r; i++) comb[i] = comb[i-1] + 1;
        }
        free(comb);
    }
    
    *output_cols = out_idx;
    free(temp);
    free(results);
}
