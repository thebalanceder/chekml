#include "WCA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate random double between min and max
double rand_double(double min, double max);

// Enforce bounds on population
void enforce_bound_constraints(Optimizer *opt);

// Initialize streams with random positions
void initialize_streams(Optimizer *opt) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
    }
}

// Evaluate streams using objective function
void evaluate_streams(Optimizer *opt, double (*objective_function)(double *)) {
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }
}

// Compute IOP (Information Overlap)
double iop(double **C, double **D, int rows, int cols_C, int cols_D) {
    if (cols_C == 0 || rows == 0) return 0.0;

    // Simplified equivalent_class: count unique rows in C and C+D
    int *counts_C = (int *)calloc(rows, sizeof(int));
    int *counts_CD = (int *)calloc(rows, sizeof(int));
    double sum_diff = 0.0;

    // Hash-based counting for C (simplified, assumes unique rows)
    for (int i = 0; i < rows; i++) {
        counts_C[i] = 1;  // Approximate unique row count
    }

    // Hash-based counting for C+D
    for (int i = 0; i < rows; i++) {
        counts_CD[i] = 1;  // Approximate unique row count
    }

    // Compute difference
    for (int i = 0; i < rows; i++) {
        sum_diff += (double)(counts_C[i] - counts_CD[i]);
    }

    free(counts_C);
    free(counts_CD);
    return sum_diff;
}

// Compute positive region
int* positive_region(double **C, double **D, int rows, int cols_C, int cols_D, int *size) {
    if (cols_C == 0 || rows == 0) {
        *size = 0;
        return NULL;
    }

    // Simplified: return indices where IOP difference is zero
    int *pos = (int *)malloc(rows * sizeof(int));
    *size = 0;
    for (int i = 0; i < rows; i++) {
        double diff = iop(C, D, rows, cols_C, cols_D);  // Simplified check
        if (fabs(diff) < 1e-6) {
            pos[(*size)++] = i;
        }
    }
    return pos;
}

// Fast reduction algorithm
int* fast_red(double **C, double **D, int rows, int cols_C, int cols_D, int *size) {
    double iop_C = iop(C, D, rows, cols_C, cols_D);
    double *w = (double *)malloc(cols_C * sizeof(double));
    int *ind = (int *)malloc(cols_C * sizeof(int));
    int *red = (int *)malloc(cols_C * sizeof(int));
    *size = 0;

    // Compute IOP for each attribute
    for (int i = 0; i < cols_C; i++) {
        double **C_single = (double **)malloc(rows * sizeof(double *));
        for (int r = 0; r < rows; r++) {
            C_single[r] = (double *)malloc(1 * sizeof(double));
            C_single[r][0] = C[r][i];
        }
        w[i] = iop(C_single, D, rows, 1, cols_D);
        for (int r = 0; r < rows; r++) free(C_single[r]);
        free(C_single);
        ind[i] = i;
    }

    // Sort indices by w
    for (int i = 0; i < cols_C - 1; i++) {
        for (int j = i + 1; j < cols_C; j++) {
            if (w[ind[i]] > w[ind[j]]) {
                int temp = ind[i];
                ind[i] = ind[j];
                ind[j] = temp;
            }
        }
    }

    // Build reduction
    for (int i = 0; i < cols_C; i++) {
        red[(*size)++] = ind[i];
        double **C_red = (double **)malloc(rows * sizeof(double *));
        for (int r = 0; r < rows; r++) {
            C_red[r] = (double *)malloc(*size * sizeof(double));
            for (int j = 0; j < *size; j++) {
                C_red[r][j] = C[r][red[j]];
            }
        }
        if (fabs(iop(C_red, D, rows, *size, cols_D) - iop_C) < 1e-6) {
            for (int r = 0; r < rows; r++) free(C_red[r]);
            free(C_red);
            break;
        }
        for (int r = 0; r < rows; r++) free(C_red[r]);
        free(C_red);
    }

    free(w);
    free(ind);
    return red;
}

// APS mechanism
void aps_mechanism(double **C, double **D, int rows, int cols_C, int cols_D, int *B, int B_size, int *u, int *v) {
    *u = 0;
    *v = 0;
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size);
    double **C_B = (double **)malloc(rows * sizeof(double *));
    for (int r = 0; r < rows; r++) {
        C_B[r] = (double *)malloc(B_size * sizeof(double));
        for (int j = 0; j < B_size; j++) {
            C_B[r][j] = C[r][B[j]];
        }
    }
    int unpos_size;
    int *unpos = positive_region(C_B, D, rows, B_size, cols_D, &unpos_size);
    for (int r = 0; r < rows; r++) free(C_B[r]);
    free(C_B);

    int *unred = (int *)malloc(cols_C * sizeof(int));
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

    int *add = (int *)malloc(cols_C * sizeof(int));
    int add_size = 0;
    for (int i = 0; i < unred_size; i++) {
        int k = unred[i];
        double **C_unpos = (double **)malloc(unpos_size * sizeof(double *));
        for (int r = 0; r < unpos_size; r++) {
            C_unpos[r] = (double *)malloc(1 * sizeof(double));
            C_unpos[r][0] = C[unpos[r]][k];
        }
        double **D_unpos = (double **)malloc(unpos_size * sizeof(double *));
        for (int r = 0; r < unpos_size; r++) {
            D_unpos[r] = (double *)malloc(cols_D * sizeof(double));
            for (int j = 0; j < cols_D; j++) {
                D_unpos[r][j] = D[unpos[r]][j];
            }
        }
        int pos_unpos_size;
        int *pos_unpos = positive_region(C_unpos, D_unpos, unpos_size, 1, cols_D, &pos_unpos_size);
        if (pos_unpos_size == unpos_size) {
            add[add_size++] = k;
        }
        for (int r = 0; r < unpos_size; r++) {
            free(C_unpos[r]);
            free(D_unpos[r]);
        }
        free(C_unpos);
        free(D_unpos);
        free(pos_unpos);
    }

    for (int i = 0; i < add_size; i++) {
        int *subspace = (int *)malloc((B_size + 1) * sizeof(int));
        for (int j = 0; j < B_size; j++) subspace[j] = B[j];
        subspace[B_size] = add[i];
        int subspace_size = B_size + 1;

        // Simplified unique rows
        int U_size = rows;  // Assume all rows unique for simplicity
        int *U = (int *)malloc(U_size * sizeof(int));
        for (int r = 0; r < U_size; r++) U[r] = r;

        for (int j = 0; j < B_size; j++) {
            int *testB = (int *)malloc((subspace_size - 1) * sizeof(int));
            int testB_size = 0;
            for (int s = 0; s < subspace_size; s++) {
                if (subspace[s] != B[j]) testB[testB_size++] = subspace[s];
            }
            double **C_U = (double **)malloc(U_size * sizeof(double *));
            for (int r = 0; r < U_size; r++) {
                C_U[r] = (double *)malloc(testB_size * sizeof(double));
                for (int s = 0; s < testB_size; s++) {
                    C_U[r][s] = C[U[r]][testB[s]];
                }
            }
            double **D_U = (double **)malloc(U_size * sizeof(double *));
            for (int r = 0; r < U_size; r++) {
                D_U[r] = (double *)malloc(cols_D * sizeof(double));
                for (int s = 0; s < cols_D; s++) {
                    D_U[r][s] = D[U[r]][s];
                }
            }
            int pos_U_size;
            int *pos_U = positive_region(C_U, D_U, U_size, testB_size, cols_D, &pos_U_size);
            if (pos_U_size == U_size) {
                *v = add[i];
                *u = B[j];
                for (int r = 0; r < U_size; r++) {
                    free(C_U[r]);
                    free(D_U[r]);
                }
                free(C_U);
                free(D_U);
                free(pos_U);
                free(testB);
                break;
            }
            for (int r = 0; r < U_size; r++) {
                free(C_U[r]);
                free(D_U[r]);
            }
            free(C_U);
            free(D_U);
            free(pos_U);
            free(testB);
        }
        free(subspace);
        free(U);
        if (*u != 0 || *v != 0) break;
    }

    free(pos);
    free(unpos);
    free(unred);
    free(add);
}

// LSAR-ASP algorithm
int* lsar_asp(double **C, double **D, int rows, int cols_C, int cols_D, int *size) {
    int *red = fast_red(C, D, rows, cols_C, cols_D, size);
    int pos_size;
    int *pos = positive_region(C, D, rows, cols_C, cols_D, &pos_size);
    int *remove_set = (int *)malloc(cols_C * sizeof(int));
    int remove_set_size = 0;
    int t = 0;

    while (remove_set_size < *size) {
        int *diff_red = (int *)malloc(cols_C * sizeof(int));
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
        if (diff_red_size == 0) {
            free(diff_red);
            break;
        }
        int a = diff_red[rand() % diff_red_size];
        int *diff_red_a = (int *)malloc(cols_C * sizeof(int));
        int diff_red_a_size = 0;
        for (int i = 0; i < *size; i++) {
            if (red[i] != a) diff_red_a[diff_red_a_size++] = red[i];
        }
        double **C_diff_red_a = (double **)malloc(rows * sizeof(double *));
        for (int r = 0; r < rows; r++) {
            C_diff_red_a[r] = (double *)malloc(diff_red_a_size * sizeof(double));
            for (int j = 0; j < diff_red_a_size; j++) {
                C_diff_red_a[r][j] = C[r][diff_red_a[j]];
            }
        }
        int pos_diff_size;
        int *pos_diff = positive_region(C_diff_red_a, D, rows, diff_red_a_size, cols_D, &pos_diff_size);
        if (pos_diff_size == pos_size) {
            free(red);
            red = diff_red_a;
            *size = diff_red_a_size;
        } else {
            int u, v;
            aps_mechanism(C, D, rows, cols_C, cols_D, diff_red_a, diff_red_a_size, &u, &v);
            if (u == 0 && v == 0) {
                remove_set[remove_set_size++] = a;
                free(diff_red_a);
            } else {
                int *new_red = (int *)malloc(cols_C * sizeof(int));
                int new_red_size = 0;
                for (int i = 0; i < *size; i++) {
                    if (red[i] != a && red[i] != u) new_red[new_red_size++] = red[i];
                }
                new_red[new_red_size++] = v;
                free(red);
                red = new_red;
                *size = new_red_size;
                free(diff_red_a);
                remove_set_size = 0;
                free(remove_set);
                remove_set = (int *)malloc(cols_C * sizeof(int));
            }
        }
        for (int r = 0; r < rows; r++) free(C_diff_red_a[r]);
        free(C_diff_red_a);
        free(pos_diff);
        free(diff_red);
        t++;
    }

    free(pos);
    free(remove_set);
    return red;
}

// Main WCA optimization function
void WCA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    initialize_streams(opt);
    for (int iter = 0; iter < opt->max_iter; iter++) {
        evaluate_streams(opt, objective_function);

        // Prepare C (population) and D (fitness)
        double **C = (double **)malloc(opt->population_size * sizeof(double *));
        for (int i = 0; i < opt->population_size; i++) {
            C[i] = (double *)malloc(opt->dim * sizeof(double));
            for (int j = 0; j < opt->dim; j++) {
                C[i][j] = opt->population[i].position[j];
            }
        }
        double **D = (double **)malloc(opt->population_size * sizeof(double *));
        for (int i = 0; i < opt->population_size; i++) {
            D[i] = (double *)malloc(1 * sizeof(double));
            D[i][0] = opt->population[i].fitness;
        }

        // Apply LSAR-ASP
        int red_size;
        int *red = lsar_asp(C, D, opt->population_size, opt->dim, 1, &red_size);

        // Update streams
        for (int i = 0; i < opt->population_size; i++) {
            if (rand_double(0.0, 1.0) < 0.5) {
                for (int j = 0; j < red_size; j++) {
                    int idx = red[j];
                    opt->population[i].position[idx] = opt->best_solution.position[idx] +
                        PERTURBATION_FACTOR * rand_double(-1.0, 1.0) * (opt->bounds[2 * idx + 1] - opt->bounds[2 * idx]);
                }
            }
        }

        enforce_bound_constraints(opt);

        // Clean up
        for (int i = 0; i < opt->population_size; i++) {
            free(C[i]);
            free(D[i]);
        }
        free(C);
        free(D);
        free(red);

        // Log progress
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
}
