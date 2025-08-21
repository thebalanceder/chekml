#include "SOA.h"
#include <time.h>
#include <limits.h>
#include <emmintrin.h> // For SSE2 intrinsics (if available)

// Define SOA_MAGIC for structure validation
#define SOA_MAGIC 0x50A50A50A50A50A5

// Helper structure to hold preallocated arrays for optimization
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
    double *random_values; // Preallocated random numbers
    int random_count;
    int random_capacity;
} SOATemps;

// Initialize SOA structure with optimized memory allocation
SOA* SOA_init(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "SOA_init: NULL optimizer\n");
        return NULL;
    }

    fprintf(stderr, "SOA_init: Starting initialization with dim=%d, population_size=%d, max_iter=%d\n",
            opt->dim, opt->population_size, opt->max_iter);
    if (opt->dim <= 0 || opt->dim > 1000 || opt->population_size <= 0 || opt->population_size > 10000 ||
        opt->max_iter <= 0 || opt->max_iter > 100000) {
        fprintf(stderr, "SOA_init: Invalid optimizer parameters: dim=%d, population_size=%d, max_iter=%d\n",
                opt->dim, opt->population_size, opt->max_iter);
        return NULL;
    }

    SOA *soa = (SOA*)calloc(1, sizeof(SOA));
    if (!soa) {
        fprintf(stderr, "SOA_init: Memory allocation failed for SOA\n");
        return NULL;
    }

    soa->magic = SOA_MAGIC;
    soa->opt = opt;
    soa->num_regions = NUM_REGIONS;
    soa->mu_max = SOA_MU_MAX;
    soa->mu_min = SOA_MU_MIN;
    soa->w_max = W_MAX_SOA;
    soa->w_min = W_MIN_SOA;

    // Allocate all arrays in one go for better locality
    size_t total_size = soa->num_regions * sizeof(int) * 3 + // start_reg, end_reg, size_reg
                        opt->dim * sizeof(double) * 2 +      // rmax, rmin
                        opt->population_size * opt->dim * sizeof(double) * 3 + // pbest_s, e_t_1, e_t_2
                        soa->num_regions * opt->dim * sizeof(double) +         // lbest_s
                        opt->population_size * sizeof(double) * 3 +            // pbest_fun, f_t_1, f_t_2
                        soa->num_regions * sizeof(double);                    // lbest_fun
    void *block = calloc(1, total_size);
    if (!block) {
        fprintf(stderr, "SOA_init: Memory allocation failed for block\n");
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
    soa->f_t_2 = (double*)ptr; ptr += opt->population_size * sizeof(double);
    soa->lbest_fun = (double*)ptr;

    // Initialize regions
    for (int r = 0; r < soa->num_regions; r++) {
        soa->start_reg[r] = (r * opt->population_size) / soa->num_regions;
        soa->end_reg[r] = ((r + 1) * opt->population_size) / soa->num_regions;
        soa->size_reg[r] = soa->end_reg[r] - soa->start_reg[r];
        fprintf(stderr, "SOA_init: Region %d: start=%d, end=%d, size=%d\n",
                r, soa->start_reg[r], soa->end_reg[r], soa->size_reg[r]);
    }

    // Initialize step sizes
    for (int j = 0; j < opt->dim; j++) {
        if (!opt->bounds) {
            fprintf(stderr, "SOA_init: NULL bounds array\n");
            SOA_free(soa);
            return NULL;
        }
        soa->rmax[j] = 0.5 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        soa->rmin[j] = -soa->rmax[j];
    }

    // Initialize population and bests
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population || !opt->population[i].position) {
            fprintf(stderr, "SOA_init: NULL population or population[%d].position\n", i);
            SOA_free(soa);
            return NULL;
        }
        for (int j = 0; j < opt->dim; j++) {
            soa->pbest_s[i * opt->dim + j] = opt->population[i].position[j];
            soa->e_t_1[i * opt->dim + j] = opt->population[i].position[j];
            soa->e_t_2[i * opt->dim + j] = opt->population[i].position[j];
        }
        soa->pbest_fun[i] = INFINITY;
        soa->f_t_1[i] = INFINITY;
        soa->f_t_2[i] = INFINITY;
    }
    for (int r = 0; r < soa->num_regions; r++) {
        for (int j = 0; j < opt->dim; j++) {
            soa->lbest_s[r * opt->dim + j] = 0.0;
        }
        soa->lbest_fun[r] = INFINITY;
    }

    fprintf(stderr, "SOA_init: Initialized SOA at %p\n", soa);
    return soa;
}

// Free SOA structure
void SOA_free(SOA *soa) {
    if (!soa) {
        fprintf(stderr, "SOA_free: NULL SOA pointer\n");
        return;
    }

    fprintf(stderr, "SOA_free: Freeing SOA at %p\n", soa);
    // Free the single memory block
    if (soa->start_reg) free(soa->start_reg); // All arrays are in one block
    free(soa);
}

// Initialize temporary arrays
static int init_temps(SOATemps *temps, int dim, int population_size) {
    size_t total_size = dim * sizeof(double) * 9 + // mu_s, x_pdirect, x_ldirect1, x_ldirect2, x_tdirect, num_pone, num_none, x_direct, r_temp
                        dim * sizeof(double) +    // en_temp
                        population_size * sizeof(double) * 10; // Random numbers buffer
    void *block = calloc(1, total_size);
    if (!block) {
        fprintf(stderr, "init_temps: Memory allocation failed\n");
        return 0;
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
    temps->random_values = (double*)ptr;
    temps->random_count = 0;
    temps->random_capacity = population_size * 10; // Enough for one generation
    return 1;
}

// Free temporary arrays
static void free_temps(SOATemps *temps) {
    if (temps->mu_s) free(temps->mu_s); // Free the single block
}

// Get random number from preallocated buffer
static double get_random(SOATemps *temps) {
    if (temps->random_count >= temps->random_capacity) {
        for (int i = 0; i < temps->random_capacity; i++) {
            temps->random_values[i] = rand() / (double)RAND_MAX;
        }
        temps->random_count = 0;
    }
    return temps->random_values[temps->random_count++];
}

// Evaluate population
static void evaluate_population(SOA *soa, ObjectiveFunction objective_function, int *fes) {
    if (!soa || !soa->opt || !objective_function || !fes) {
        fprintf(stderr, "evaluate_population: Invalid arguments\n");
        return;
    }
    if (soa->magic != SOA_MAGIC) {
        fprintf(stderr, "evaluate_population: Invalid SOA magic number\n");
        return;
    }

    Optimizer *opt = soa->opt;
    if (opt->population_size <= 0 || opt->dim <= 0 || opt->dim > 1000) {
        fprintf(stderr, "evaluate_population: Invalid population_size=%d or dim=%d\n",
                opt->population_size, opt->dim);
        return;
    }

    fprintf(stderr, "evaluate_population: Evaluating %d solutions\n", opt->population_size);
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population || !opt->population[i].position) {
            fprintf(stderr, "evaluate_population: NULL population or population[%d].position\n", i);
            continue;
        }
        double fitness = objective_function(opt->population[i].position);
        if (isnan(fitness) || isinf(fitness)) {
            fprintf(stderr, "evaluate_population: Invalid fitness for solution %d\n", i);
            continue;
        }
        opt->population[i].fitness = fitness;
        (*fes)++;
        fprintf(stderr, "evaluate_population: Solution %d fitness=%f\n", i, opt->population[i].fitness);

        if (opt->population[i].fitness < soa->pbest_fun[i]) {
            soa->pbest_fun[i] = opt->population[i].fitness;
            memcpy(soa->pbest_s + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        }

        if (!opt->best_solution.position) {
            fprintf(stderr, "evaluate_population: NULL best_solution.position\n");
            continue;
        }
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            fprintf(stderr, "evaluate_population: New global best fitness=%f\n", opt->best_solution.fitness);
        }
    }
    enforce_bound_constraints(opt);
    fprintf(stderr, "evaluate_population: Completed evaluation\n");
}

// Update region bests
static void update_region_bests(SOA *soa) {
    if (!soa || !soa->opt) {
        fprintf(stderr, "update_region_bests: Invalid SOA or optimizer\n");
        return;
    }
    if (soa->magic != SOA_MAGIC) {
        fprintf(stderr, "update_region_bests: Invalid SOA magic number\n");
        return;
    }

    Optimizer *opt = soa->opt;
    fprintf(stderr, "update_region_bests: Updating %d regions\n", soa->num_regions);
    for (int r = 0; r < soa->num_regions; r++) {
        if (soa->size_reg[r] <= 0) {
            fprintf(stderr, "update_region_bests: Invalid region %d size=%d\n", r, soa->size_reg[r]);
            continue;
        }
        int best_idx = soa->start_reg[r];
        double best_fun = opt->population[best_idx].fitness;
        for (int i = soa->start_reg[r]; i < soa->end_reg[r]; i++) {
            if (!opt->population || !opt->population[i].position) {
                fprintf(stderr, "update_region_bests: NULL population or population[%d].position\n", i);
                continue;
            }
            if (opt->population[i].fitness < best_fun) {
                best_fun = opt->population[i].fitness;
                best_idx = i;
            }
        }
        if (best_fun < soa->lbest_fun[r]) {
            soa->lbest_fun[r] = best_fun;
            memcpy(soa->lbest_s + r * opt->dim, opt->population[best_idx].position, opt->dim * sizeof(double));
            fprintf(stderr, "update_region_bests: Region %d new best fitness=%f\n", r, best_fun);
        }
    }
    fprintf(stderr, "update_region_bests: Completed\n");
}

// Main optimization function
void SOA_optimize(void *soa_ptr, ObjectiveFunction objective_function) {
    fprintf(stderr, "SOA_optimize: Function entered with soa_ptr=%p\n", soa_ptr);
    if (!soa_ptr || !objective_function) {
        fprintf(stderr, "SOA_optimize: Invalid arguments: soa_ptr=%p, objective_function=%p\n",
                soa_ptr, objective_function);
        return;
    }

    SOA *soa = (SOA*)soa_ptr;
    fprintf(stderr, "SOA_optimize: Cast soa_ptr to soa=%p\n", soa);

    // Check if soa_ptr is actually an Optimizer pointer
    Optimizer *opt = (Optimizer*)soa_ptr;
    if (soa->magic != SOA_MAGIC) {
        fprintf(stderr, "SOA_optimize: Invalid SOA magic number: expected=%lx, got=%lx\n",
                SOA_MAGIC, soa->magic);
        fprintf(stderr, "SOA_optimize: Assuming soa_ptr is Optimizer pointer, initializing SOA\n");

        // Validate Optimizer fields
        if (!opt->population || !opt->bounds || !opt->best_solution.position ||
            opt->dim <= 0 || opt->dim > 1000 ||
            opt->population_size <= 0 || opt->population_size > 10000 ||
            opt->max_iter <= 0 || opt->max_iter > 100000) {
            fprintf(stderr, "SOA_optimize: Invalid Optimizer parameters: dim=%d, population_size=%d, max_iter=%d\n",
                    opt->dim, opt->population_size, opt->max_iter);
            return;
        }

        // Create temporary SOA structure
        soa = SOA_init(opt);
        if (!soa) {
            fprintf(stderr, "SOA_optimize: Failed to initialize SOA\n");
            return;
        }
        fprintf(stderr, "SOA_optimize: Created temporary SOA at %p\n", soa);
    } else {
        fprintf(stderr, "SOA_optimize: Valid SOA magic number\n");
        if (!soa->opt) {
            fprintf(stderr, "SOA_optimize: NULL optimizer in soa\n");
            return;
        }
        opt = soa->opt;
    }

    fprintf(stderr, "SOA_optimize: Starting optimization with dim=%d, population_size=%d, max_iter=%d\n",
            opt->dim, opt->population_size, opt->max_iter);

    // Initialize temporary arrays
    SOATemps temps = {0};
    if (!init_temps(&temps, opt->dim, opt->population_size)) {
        if (soa->magic != SOA_MAGIC) SOA_free(soa);
        return;
    }

    // Preallocate sorted indices for all regions
    int **sorted_indices = (int**)calloc(soa->num_regions, sizeof(int*));
    int *indices_changed = (int*)calloc(soa->num_regions, sizeof(int));
    if (!sorted_indices || !indices_changed) {
        fprintf(stderr, "SOA_optimize: Memory allocation failed for sorted_indices or indices_changed\n");
        free_temps(&temps);
        if (soa->magic != SOA_MAGIC) SOA_free(soa);
        free(sorted_indices);
        free(indices_changed);
        return;
    }
    for (int r = 0; r < soa->num_regions; r++) {
        sorted_indices[r] = (int*)calloc(soa->size_reg[r], sizeof(int));
        if (!sorted_indices[r]) {
            fprintf(stderr, "SOA_optimize: Memory allocation failed for sorted_indices[%d]\n", r);
            for (int i = 0; i < r; i++) free(sorted_indices[i]);
            free(sorted_indices);
            free(indices_changed);
            free_temps(&temps);
            if (soa->magic != SOA_MAGIC) SOA_free(soa);
            return;
        }
        for (int i = 0; i < soa->size_reg[r]; i++) {
            sorted_indices[r][i] = soa->start_reg[r] + i;
        }
        indices_changed[r] = 1; // Force initial sort
    }

    int max_fes = opt->max_iter * opt->population_size;
    int max_gens = opt->max_iter;
    int fes = 0, gens = 0;
    double error_prev = INFINITY;

    // Initial evaluation
    fprintf(stderr, "SOA_optimize: Initial population evaluation\n");
    evaluate_population(soa, objective_function, &fes);
    update_region_bests(soa);

    // Preallocate temporary population
    double *temp_population = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
    if (!temp_population) {
        fprintf(stderr, "SOA_optimize: Memory allocation failed for temp_population\n");
        for (int r = 0; r < soa->num_regions; r++) free(sorted_indices[r]);
        free(sorted_indices);
        free(indices_changed);
        free_temps(&temps);
        if (soa->magic != SOA_MAGIC) SOA_free(soa);
        return;
    }

    while (fes < max_fes) {
        gens++;
        double weight = soa->w_max - gens * (soa->w_max - soa->w_min) / max_gens;
        double mu = soa->mu_max - gens * (soa->mu_max - soa->mu_min) / max_gens;
        fprintf(stderr, "SOA_optimize: Generation %d, weight=%f, mu=%f\n", gens, weight, mu);

        // Copy population to temp_population
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(temp_population + i * opt->dim, opt->population[i].position, opt->dim * sizeof(double));
        }

        for (int r = 0; r < soa->num_regions; r++) {
            fprintf(stderr, "SOA_optimize: Processing region %d\n", r);
            if (soa->size_reg[r] <= 0) {
                fprintf(stderr, "SOA_optimize: Invalid region %d size=%d\n", r, soa->size_reg[r]);
                continue;
            }

            // Sort indices by fitness (descending) only if changed
            if (indices_changed[r]) {
                for (int i = 0; i < soa->size_reg[r] - 1; i++) {
                    for (int j = 0; j < soa->size_reg[r] - i - 1; j++) {
                        int idx1 = sorted_indices[r][j], idx2 = sorted_indices[r][j + 1];
                        if (opt->population[idx1].fitness < opt->population[idx2].fitness) {
                            int temp = sorted_indices[r][j];
                            sorted_indices[r][j] = sorted_indices[r][j + 1];
                            sorted_indices[r][j + 1] = temp;
                        }
                    }
                }
                indices_changed[r] = 0;
            }

            // Compute exploration term
            if (soa->size_reg[r] < 2) {
                fprintf(stderr, "SOA_optimize: Region %d too small (size=%d)\n", r, soa->size_reg[r]);
                continue;
            }
            int rand_en = 1 + (int)(get_random(&temps) * (soa->size_reg[r] - 2));
            if (rand_en >= soa->size_reg[r]) rand_en = soa->size_reg[r] - 1;
            for (int j = 0; j < opt->dim; j++) {
                temps.en_temp[j] = weight * fabs(opt->population[sorted_indices[r][soa->size_reg[r] - 1]].position[j] - 
                                                opt->population[sorted_indices[r][rand_en]].position[j]);
            }

            for (int s = soa->start_reg[r]; s < soa->end_reg[r]; s++) {
                fprintf(stderr, "SOA_optimize: Updating solution %d in region %d\n", s, r);
                for (int j = 0; j < opt->dim; j++) {
                    temps.mu_s[j] = mu + (1.0 - mu) * get_random(&temps);
                    if (temps.mu_s[j] <= 0.0 || temps.mu_s[j] >= 1.0) temps.mu_s[j] = mu;
                }

                // Compute directions
                for (int j = 0; j < opt->dim; j++) {
                    temps.x_pdirect[j] = soa->pbest_s[s * opt->dim + j] - opt->population[s].position[j];
                    temps.x_ldirect1[j] = (soa->lbest_fun[r] < opt->population[s].fitness) ? 
                                         soa->lbest_s[r * opt->dim + j] - opt->population[s].position[j] : 0.0;
                    temps.x_ldirect2[j] = (opt->population[sorted_indices[r][soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 
                                         opt->population[sorted_indices[r][soa->size_reg[r] - 1]].position[j] - opt->population[s].position[j] : 0.0;
                }

                // Temporal direction
                double f_values[3] = {soa->f_t_2[s], soa->f_t_1[s], opt->population[s].fitness};
                double *e_values[3] = {soa->e_t_2 + s * opt->dim, soa->e_t_1 + s * opt->dim, opt->population[s].position};
                int order_idx[3] = {0, 1, 2};
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2 - i; j++) {
                        if (f_values[order_idx[j]] > f_values[order_idx[j + 1]]) {
                            int temp = order_idx[j];
                            order_idx[j] = order_idx[j + 1];
                            order_idx[j + 1] = temp;
                        }
                    }
                }
                for (int j = 0; j < opt->dim; j++) {
                    temps.x_tdirect[j] = e_values[order_idx[0]][j] - e_values[order_idx[2]][j];
                }

                // Compute direction signs
                int flag_direct[4] = {1, 1, (soa->lbest_fun[r] < opt->population[s].fitness) ? 1 : 0, 
                                     (opt->population[sorted_indices[r][soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 1 : 0};
                double x_signs[4][opt->dim];
                for (int j = 0; j < opt->dim; j++) {
                    x_signs[0][j] = (temps.x_tdirect[j] > 0) ? 1 : (temps.x_tdirect[j] < 0) ? -1 : 0;
                    x_signs[1][j] = (temps.x_pdirect[j] > 0) ? 1 : (temps.x_pdirect[j] < 0) ? -1 : 0;
                    x_signs[2][j] = (temps.x_ldirect1[j] > 0) ? 1 : (temps.x_ldirect1[j] < 0) ? -1 : 0;
                    x_signs[3][j] = (temps.x_ldirect2[j] > 0) ? 1 : (temps.x_ldirect2[j] < 0) ? -1 : 0;
                }

                int select_sign[4], num_sign = 0;
                for (int i = 0; i < 4; i++) {
                    if (flag_direct[i] > 0) select_sign[num_sign++] = i;
                }

                memset(temps.num_pone, 0, opt->dim * sizeof(double));
                memset(temps.num_none, 0, opt->dim * sizeof(double));
                for (int i = 0; i < num_sign; i++) {
                    for (int j = 0; j < opt->dim; j++) {
                        temps.num_pone[j] += (fabs(x_signs[select_sign[i]][j]) + x_signs[select_sign[i]][j]) / 2.0;
                        temps.num_none[j] += (fabs(x_signs[select_sign[i]][j]) - x_signs[select_sign[i]][j]) / 2.0;
                    }
                }

                for (int j = 0; j < opt->dim; j++) {
                    double prob_pone = num_sign > 0 ? temps.num_pone[j] / num_sign : 0.5;
                    double prob_none = num_sign > 0 ? (temps.num_pone[j] + temps.num_none[j]) / num_sign : 1.0;
                    double rand_roulette = get_random(&temps);
                    temps.x_direct[j] = (rand_roulette <= prob_pone) ? 1.0 : (rand_roulette <= prob_none) ? -1.0 : 0.0;
                }

                // Adjust direction based on bounds
                for (int j = 0; j < opt->dim; j++) {
                    if (opt->population[s].position[j] > opt->bounds[2 * j + 1]) temps.x_direct[j] = -1.0;
                    if (opt->population[s].position[j] < opt->bounds[2 * j]) temps.x_direct[j] = 1.0;
                    if (temps.x_direct[j] == 0.0) {
                        temps.x_direct[j] = (get_random(&temps) < 0.5) ? 1.0 : -1.0;
                    }
                }

                // Compute step (vectorized for small dim)
                for (int j = 0; j < opt->dim; j++) {
                    temps.r_temp[j] = temps.x_direct[j] * (temps.en_temp[j] * sqrt(-2.0 * log(temps.mu_s[j])));
                    temps.r_temp[j] = fmax(soa->rmin[j], fmin(soa->rmax[j], temps.r_temp[j]));
                    temp_population[s * opt->dim + j] = opt->population[s].position[j] + temps.r_temp[j];
                }
            }
        }

        // Evaluate offspring
        fprintf(stderr, "SOA_optimize: Evaluating offspring\n");
        for (int s = 0; s < opt->population_size; s++) {
            double temp_val = objective_function(&temp_population[s * opt->dim]);
            fes++;
            indices_changed[s / (opt->population_size / soa->num_regions)] = 1; // Mark region as changed

            // Update history
            memcpy(soa->e_t_2 + s * opt->dim, soa->e_t_1 + s * opt->dim, opt->dim * sizeof(double));
            memcpy(soa->e_t_1 + s * opt->dim, opt->population[s].position, opt->dim * sizeof(double));
            memcpy(opt->population[s].position, temp_population + s * opt->dim, opt->dim * sizeof(double));
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
                fprintf(stderr, "SOA_optimize: New global best fitness=%f\n", temp_val);
            }
        }

        // Shuffle population if no improvement
        if (error_prev <= opt->best_solution.fitness || gens % 1 == 0) {
            fprintf(stderr, "SOA_optimize: Shuffling population\n");
            int *perm = (int*)calloc(opt->population_size, sizeof(int));
            if (!perm) {
                fprintf(stderr, "SOA_optimize: Memory allocation failed for perm\n");
                free(temp_population);
                for (int r = 0; r < soa->num_regions; r++) free(sorted_indices[r]);
                free(sorted_indices);
                free(indices_changed);
                free_temps(&temps);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            for (int i = 0; i < opt->population_size; i++) perm[i] = i;
            for (int i = opt->population_size - 1; i > 0; i--) {
                int j = (int)(get_random(&temps) * (i + 1));
                int temp = perm[i];
                perm[i] = perm[j];
                perm[j] = temp;
            }

            double *temp_pop = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
            double *temp_fit = (double*)calloc(opt->population_size, sizeof(double));
            double *temp_pbest_s = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
            double *temp_pbest_fun = (double*)calloc(opt->population_size, sizeof(double));
            double *temp_e_t_1 = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
            double *temp_e_t_2 = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
            double *temp_f_t_1 = (double*)calloc(opt->population_size, sizeof(double));
            double *temp_f_t_2 = (double*)calloc(opt->population_size, sizeof(double));
            if (!temp_pop || !temp_fit || !temp_pbest_s || !temp_pbest_fun ||
                !temp_e_t_1 || !temp_e_t_2 || !temp_f_t_1 || !temp_f_t_2) {
                fprintf(stderr, "SOA_optimize: Memory allocation failed for shuffle arrays\n");
                free(perm);
                free(temp_pop);
                free(temp_fit);
                free(temp_pbest_s);
                free(temp_pbest_fun);
                free(temp_e_t_1);
                free(temp_e_t_2);
                free(temp_f_t_1);
                free(temp_f_t_2);
                free(temp_population);
                for (int r = 0; r < soa->num_regions; r++) free(sorted_indices[r]);
                free(sorted_indices);
                free(indices_changed);
                free_temps(&temps);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }

            for (int i = 0; i < opt->population_size; i++) {
                int idx = perm[i];
                memcpy(temp_pop + i * opt->dim, opt->population[idx].position, opt->dim * sizeof(double));
                memcpy(temp_pbest_s + i * opt->dim, soa->pbest_s + idx * opt->dim, opt->dim * sizeof(double));
                memcpy(temp_e_t_1 + i * opt->dim, soa->e_t_1 + idx * opt->dim, opt->dim * sizeof(double));
                memcpy(temp_e_t_2 + i * opt->dim, soa->e_t_2 + idx * opt->dim, opt->dim * sizeof(double));
                temp_fit[i] = opt->population[idx].fitness;
                temp_pbest_fun[i] = soa->pbest_fun[idx];
                temp_f_t_1[i] = soa->f_t_1[idx];
                temp_f_t_2[i] = soa->f_t_2[idx];
            }

            for (int i = 0; i < opt->population_size; i++) {
                memcpy(opt->population[i].position, temp_pop + i * opt->dim, opt->dim * sizeof(double));
                memcpy(soa->pbest_s + i * opt->dim, temp_pbest_s + i * opt->dim, opt->dim * sizeof(double));
                memcpy(soa->e_t_1 + i * opt->dim, temp_e_t_1 + i * opt->dim, opt->dim * sizeof(double));
                memcpy(soa->e_t_2 + i * opt->dim, temp_e_t_2 + i * opt->dim, opt->dim * sizeof(double));
                opt->population[i].fitness = temp_fit[i];
                soa->pbest_fun[i] = temp_pbest_fun[i];
                soa->f_t_1[i] = temp_f_t_1[i];
                soa->f_t_2[i] = temp_f_t_2[i];
            }

            free(temp_pop);
            free(temp_fit);
            free(temp_pbest_s);
            free(temp_pbest_fun);
            free(temp_e_t_1);
            free(temp_e_t_2);
            free(temp_f_t_1);
            free(temp_f_t_2);
            free(perm);
            for (int r = 0; r < soa->num_regions; r++) indices_changed[r] = 1; // Resort after shuffle
        }

        error_prev = opt->best_solution.fitness;
        update_region_bests(soa);
        enforce_bound_constraints(opt);
    }

    // Cleanup
    free(temp_population);
    for (int r = 0; r < soa->num_regions; r++) free(sorted_indices[r]);
    free(sorted_indices);
    free(indices_changed);
    free_temps(&temps);
    fprintf(stderr, "SOA_optimize: Optimization completed\n");
    if (soa->magic != SOA_MAGIC) {
        fprintf(stderr, "SOA_optimize: Freeing temporary SOA at %p\n", soa);
        SOA_free(soa);
    }
}
