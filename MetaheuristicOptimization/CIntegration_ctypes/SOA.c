#include "SOA.h"
#include <time.h>
#include <limits.h>

#define SOA_MAGIC 0x50A50A50A50A50A5 // Unique identifier for SOA structure

// Initialize SOA structure
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

    // Initialize regions
    soa->start_reg = (int*)calloc(soa->num_regions, sizeof(int));
    soa->end_reg = (int*)calloc(soa->num_regions, sizeof(int));
    soa->size_reg = (int*)calloc(soa->num_regions, sizeof(int));
    if (!soa->start_reg || !soa->end_reg || !soa->size_reg) {
        fprintf(stderr, "SOA_init: Memory allocation failed for region arrays\n");
        SOA_free(soa);
        return NULL;
    }
    for (int r = 0; r < soa->num_regions; r++) {
        soa->start_reg[r] = (r * opt->population_size) / soa->num_regions;
        soa->end_reg[r] = ((r + 1) * opt->population_size) / soa->num_regions;
        soa->size_reg[r] = soa->end_reg[r] - soa->start_reg[r];
        fprintf(stderr, "SOA_init: Region %d: start=%d, end=%d, size=%d\n",
                r, soa->start_reg[r], soa->end_reg[r], soa->size_reg[r]);
    }

    // Initialize step sizes
    soa->rmax = (double*)calloc(opt->dim, sizeof(double));
    soa->rmin = (double*)calloc(opt->dim, sizeof(double));
    if (!soa->rmax || !soa->rmin) {
        fprintf(stderr, "SOA_init: Memory allocation failed for rmax/rmin\n");
        SOA_free(soa);
        return NULL;
    }
    for (int j = 0; j < opt->dim; j++) {
        if (!opt->bounds) {
            fprintf(stderr, "SOA_init: NULL bounds array\n");
            SOA_free(soa);
            return NULL;
        }
        soa->rmax[j] = 0.5 * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        soa->rmin[j] = -soa->rmax[j];
    }

    // Initialize personal and local bests
    soa->pbest_s = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
    soa->pbest_fun = (double*)calloc(opt->population_size, sizeof(double));
    soa->lbest_s = (double*)calloc(soa->num_regions * opt->dim, sizeof(double));
    soa->lbest_fun = (double*)calloc(soa->num_regions, sizeof(double));
    soa->e_t_1 = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
    soa->e_t_2 = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
    soa->f_t_1 = (double*)calloc(opt->population_size, sizeof(double));
    soa->f_t_2 = (double*)calloc(opt->population_size, sizeof(double));
    if (!soa->pbest_s || !soa->pbest_fun || !soa->lbest_s || !soa->lbest_fun ||
        !soa->e_t_1 || !soa->e_t_2 || !soa->f_t_1 || !soa->f_t_2) {
        fprintf(stderr, "SOA_init: Memory allocation failed for best/history arrays\n");
        SOA_free(soa);
        return NULL;
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
    if (soa->start_reg) free(soa->start_reg);
    if (soa->end_reg) free(soa->end_reg);
    if (soa->size_reg) free(soa->size_reg);
    if (soa->rmax) free(soa->rmax);
    if (soa->rmin) free(soa->rmin);
    if (soa->pbest_s) free(soa->pbest_s);
    if (soa->pbest_fun) free(soa->pbest_fun);
    if (soa->lbest_s) free(soa->lbest_s);
    if (soa->lbest_fun) free(soa->lbest_fun);
    if (soa->e_t_1) free(soa->e_t_1);
    if (soa->e_t_2) free(soa->e_t_2);
    if (soa->f_t_1) free(soa->f_t_1);
    if (soa->f_t_2) free(soa->f_t_2);
    free(soa);
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
            for (int j = 0; j < opt->dim; j++) {
                soa->pbest_s[i * opt->dim + j] = opt->population[i].position[j];
            }
        }

        if (!opt->best_solution.position) {
            fprintf(stderr, "evaluate_population: NULL best_solution.position\n");
            continue;
        }
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[i].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
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
    //fprintf(stderr, "update_region_bests: Updating %d regions\n", soa->num_regions);
    for (int r = 0; r < soa->num_regions; r++) {
        if (soa->start_reg[r] >= opt->population_size || soa->end_reg[r] > opt->population_size ||
            soa->size_reg[r] <= 0) {
            fprintf(stderr, "update_region_bests: Invalid region %d indices: start=%d, end=%d, size=%d\n",
                    r, soa->start_reg[r], soa->end_reg[r], soa->size_reg[r]);
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
            for (int j = 0; j < opt->dim; j++) {
                soa->lbest_s[r * opt->dim + j] = opt->population[best_idx].position[j];
            }
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

    int max_fes = opt->max_iter * opt->population_size;
    int max_gens = opt->max_iter;
    int fes = 0, gens = 0;
    double error_prev = INFINITY;

    // Initial evaluation
    fprintf(stderr, "SOA_optimize: Initial population evaluation\n");
    evaluate_population(soa, objective_function, &fes);
    update_region_bests(soa);

    while (fes < max_fes) {
        gens++;
        double weight = soa->w_max - gens * (soa->w_max - soa->w_min) / max_gens;
        double mu = soa->mu_max - gens * (soa->mu_max - soa->mu_min) / max_gens;
        fprintf(stderr, "SOA_optimize: Generation %d, weight=%f, mu=%f\n", gens, weight, mu);

        // Temporary population
        double *temp_population = (double*)calloc(opt->population_size * opt->dim, sizeof(double));
        if (!temp_population) {
            fprintf(stderr, "SOA_optimize: Memory allocation failed for temp_population\n");
            if (soa->magic != SOA_MAGIC) SOA_free(soa); // Free temporary SOA
            return;
        }
        for (int i = 0; i < opt->population_size; i++) {
            if (!opt->population || !opt->population[i].position) {
                fprintf(stderr, "SOA_optimize: NULL population or population[%d].position\n", i);
                free(temp_population);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            for (int j = 0; j < opt->dim; j++) {
                temp_population[i * opt->dim + j] = opt->population[i].position[j];
            }
        }

        for (int r = 0; r < soa->num_regions; r++) {
            fprintf(stderr, "SOA_optimize: Processing region %d\n", r);
            if (soa->size_reg[r] <= 0) {
                fprintf(stderr, "SOA_optimize: Invalid region %d size=%d\n", r, soa->size_reg[r]);
                continue;
            }
            // Sort indices by fitness (descending)
            int *sorted_indices = (int*)calloc(soa->size_reg[r], sizeof(int));
            if (!sorted_indices) {
                fprintf(stderr, "SOA_optimize: Memory allocation failed for sorted_indices\n");
                free(temp_population);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            for (int i = 0; i < soa->size_reg[r]; i++) {
                sorted_indices[i] = soa->start_reg[r] + i;
            }
            for (int i = 0; i < soa->size_reg[r] - 1; i++) {
                for (int j = 0; j < soa->size_reg[r] - i - 1; j++) {
                    int idx1 = sorted_indices[j], idx2 = sorted_indices[j + 1];
                    if (!opt->population[idx1].position || !opt->population[idx2].position) {
                        fprintf(stderr, "SOA_optimize: NULL population position at idx1=%d or idx2=%d\n", idx1, idx2);
                        continue;
                    }
                    if (opt->population[idx1].fitness < opt->population[idx2].fitness) {
                        int temp = sorted_indices[j];
                        sorted_indices[j] = sorted_indices[j + 1];
                        sorted_indices[j + 1] = temp;
                    }
                }
            }

            // Compute exploration term
            if (soa->size_reg[r] < 2) {
                fprintf(stderr, "SOA_optimize: Region %d too small (size=%d)\n", r, soa->size_reg[r]);
                free(sorted_indices);
                continue;
            }
            int rand_en = 1 + (int)((rand() / (double)RAND_MAX) * (soa->size_reg[r] - 2));
            if (rand_en >= soa->size_reg[r]) rand_en = soa->size_reg[r] - 1;
            double *en_temp = (double*)calloc(opt->dim, sizeof(double));
            if (!en_temp) {
                fprintf(stderr, "SOA_optimize: Memory allocation failed for en_temp\n");
                free(sorted_indices);
                free(temp_population);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            for (int j = 0; j < opt->dim; j++) {
                if (!opt->population[sorted_indices[soa->size_reg[r] - 1]].position ||
                    !opt->population[sorted_indices[rand_en]].position) {
                    fprintf(stderr, "SOA_optimize: NULL position in region %d at indices %d or %d\n",
                            r, sorted_indices[soa->size_reg[r] - 1], sorted_indices[rand_en]);
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                en_temp[j] = weight * fabs(opt->population[sorted_indices[soa->size_reg[r] - 1]].position[j] - 
                                          opt->population[sorted_indices[rand_en]].position[j]);
            }

            for (int s = soa->start_reg[r]; s < soa->end_reg[r]; s++) {
                //fprintf(stderr, "SOA_optimize: Updating solution %d in region %d\n", s, r);
                double *mu_s = (double*)calloc(opt->dim, sizeof(double));
                if (!mu_s) {
                    fprintf(stderr, "SOA_optimize: Memory allocation failed for mu_s\n");
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                for (int j = 0; j < opt->dim; j++) {
                    mu_s[j] = mu + (1.0 - mu) * (rand() / (double)RAND_MAX);
                    if (mu_s[j] <= 0.0 || mu_s[j] >= 1.0) mu_s[j] = mu; // Prevent log(0)
                }

                // Compute directions
                double *x_pdirect = (double*)calloc(opt->dim, sizeof(double));
                double *x_ldirect1 = (double*)calloc(opt->dim, sizeof(double));
                double *x_ldirect2 = (double*)calloc(opt->dim, sizeof(double));
                double *x_tdirect = (double*)calloc(opt->dim, sizeof(double));
                if (!x_pdirect || !x_ldirect1 || !x_ldirect2 || !x_tdirect) {
                    fprintf(stderr, "SOA_optimize: Memory allocation failed for direction arrays\n");
                    free(mu_s);
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (x_pdirect) free(x_pdirect);
                    if (x_ldirect1) free(x_ldirect1);
                    if (x_ldirect2) free(x_ldirect2);
                    if (x_tdirect) free(x_tdirect);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }

                for (int j = 0; j < opt->dim; j++) {
                    if (!opt->population[s].position) {
                        fprintf(stderr, "SOA_optimize: NULL population[%d].position\n", s);
                        free(x_tdirect);
                        free(x_ldirect2);
                        free(x_ldirect1);
                        free(x_pdirect);
                        free(mu_s);
                        free(en_temp);
                        free(sorted_indices);
                        free(temp_population);
                        if (soa->magic != SOA_MAGIC) SOA_free(soa);
                        return;
                    }
                    x_pdirect[j] = soa->pbest_s[s * opt->dim + j] - opt->population[s].position[j];
                    x_ldirect1[j] = (soa->lbest_fun[r] < opt->population[s].fitness) ? 
                                    soa->lbest_s[r * opt->dim + j] - opt->population[s].position[j] : 0.0;
                    x_ldirect2[j] = (opt->population[sorted_indices[soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 
                                    opt->population[sorted_indices[soa->size_reg[r] - 1]].position[j] - opt->population[s].position[j] : 0.0;
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
                    x_tdirect[j] = e_values[order_idx[0]][j] - e_values[order_idx[2]][j];
                }

                // Compute direction signs
                int flag_direct[4] = {1, 1, (soa->lbest_fun[r] < opt->population[s].fitness) ? 1 : 0, 
                                     (opt->population[sorted_indices[soa->size_reg[r] - 1]].fitness < opt->population[s].fitness) ? 1 : 0};
                double x_signs[4][opt->dim];
                for (int j = 0; j < opt->dim; j++) {
                    x_signs[0][j] = (x_tdirect[j] > 0) ? 1 : (x_tdirect[j] < 0) ? -1 : 0;
                    x_signs[1][j] = (x_pdirect[j] > 0) ? 1 : (x_pdirect[j] < 0) ? -1 : 0;
                    x_signs[2][j] = (x_ldirect1[j] > 0) ? 1 : (x_ldirect1[j] < 0) ? -1 : 0;
                    x_signs[3][j] = (x_ldirect2[j] > 0) ? 1 : (x_ldirect2[j] < 0) ? -1 : 0;
                }

                int select_sign[4], num_sign = 0;
                for (int i = 0; i < 4; i++) {
                    if (flag_direct[i] > 0) select_sign[num_sign++] = i;
                }

                double *num_pone = (double*)calloc(opt->dim, sizeof(double));
                double *num_none = (double*)calloc(opt->dim, sizeof(double));
                if (!num_pone || !num_none) {
                    fprintf(stderr, "SOA_optimize: Memory allocation failed for num_pone/num_none\n");
                    free(x_tdirect);
                    free(x_ldirect2);
                    free(x_ldirect1);
                    free(x_pdirect);
                    free(mu_s);
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (num_pone) free(num_pone);
                    if (num_none) free(num_none);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                for (int i = 0; i < num_sign; i++) {
                    for (int j = 0; j < opt->dim; j++) {
                        num_pone[j] += (fabs(x_signs[select_sign[i]][j]) + x_signs[select_sign[i]][j]) / 2.0;
                        num_none[j] += (fabs(x_signs[select_sign[i]][j]) - x_signs[select_sign[i]][j]) / 2.0;
                    }
                }

                double *x_direct = (double*)calloc(opt->dim, sizeof(double));
                if (!x_direct) {
                    fprintf(stderr, "SOA_optimize: Memory allocation failed for x_direct\n");
                    free(num_none);
                    free(num_pone);
                    free(x_tdirect);
                    free(x_ldirect2);
                    free(x_ldirect1);
                    free(x_pdirect);
                    free(mu_s);
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                for (int j = 0; j < opt->dim; j++) {
                    double prob_pone = num_sign > 0 ? num_pone[j] / num_sign : 0.5;
                    double prob_none = num_sign > 0 ? (num_pone[j] + num_none[j]) / num_sign : 1.0;
                    double rand_roulette = rand() / (double)RAND_MAX;
                    x_direct[j] = (rand_roulette <= prob_pone) ? 1.0 : (rand_roulette <= prob_none) ? -1.0 : 0.0;
                }

                // Adjust direction based on bounds
                for (int j = 0; j < opt->dim; j++) {
                    if (!opt->bounds) {
                        fprintf(stderr, "SOA_optimize: NULL bounds array\n");
                        free(x_direct);
                        free(num_none);
                        free(num_pone);
                        free(x_tdirect);
                        free(x_ldirect2);
                        free(x_ldirect1);
                        free(x_pdirect);
                        free(mu_s);
                        free(en_temp);
                        free(sorted_indices);
                        free(temp_population);
                        if (soa->magic != SOA_MAGIC) SOA_free(soa);
                        return;
                    }
                    if (opt->population[s].position[j] > opt->bounds[2 * j + 1]) x_direct[j] = -1.0;
                    if (opt->population[s].position[j] < opt->bounds[2 * j]) x_direct[j] = 1.0;
                }
                for (int j = 0; j < opt->dim; j++) {
                    if (x_direct[j] == 0.0) {
                        x_direct[j] = (rand() / (double)RAND_MAX < 0.5) ? 1.0 : -1.0;
                    }
                }

                // Compute step
                double *r_temp = (double*)calloc(opt->dim, sizeof(double));
                if (!r_temp) {
                    fprintf(stderr, "SOA_optimize: Memory allocation failed for r_temp\n");
                    free(x_direct);
                    free(num_none);
                    free(num_pone);
                    free(x_tdirect);
                    free(x_ldirect2);
                    free(x_ldirect1);
                    free(x_pdirect);
                    free(mu_s);
                    free(en_temp);
                    free(sorted_indices);
                    free(temp_population);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                for (int j = 0; j < opt->dim; j++) {
                    r_temp[j] = x_direct[j] * (en_temp[j] * sqrt(-2.0 * log(mu_s[j])));
                    r_temp[j] = fmax(soa->rmin[j], fmin(soa->rmax[j], r_temp[j]));
                }

                // Update offspring
                for (int j = 0; j < opt->dim; j++) {
                    temp_population[s * opt->dim + j] = opt->population[s].position[j] + r_temp[j];
                }

                free(r_temp);
                free(x_direct);
                free(num_none);
                free(num_pone);
                free(x_tdirect);
                free(x_ldirect2);
                free(x_ldirect1);
                free(x_pdirect);
                free(mu_s);
            }

            free(en_temp);
            free(sorted_indices);
        }

        // Evaluate offspring
        fprintf(stderr, "SOA_optimize: Evaluating offspring\n");
        for (int s = 0; s < opt->population_size; s++) {
            if (!opt->population || !opt->population[s].position) {
                fprintf(stderr, "SOA_optimize: NULL population or population[%d].position\n", s);
                free(temp_population);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            double temp_val = objective_function(&temp_population[s * opt->dim]);
            fes++;

            // Update history
            for (int j = 0; j < opt->dim; j++) {
                soa->e_t_2[s * opt->dim + j] = soa->e_t_1[s * opt->dim + j];
                soa->e_t_1[s * opt->dim + j] = opt->population[s].position[j];
                opt->population[s].position[j] = temp_population[s * opt->dim + j];
            }
            soa->f_t_2[s] = soa->f_t_1[s];
            soa->f_t_1[s] = opt->population[s].fitness;
            opt->population[s].fitness = temp_val;

            if (temp_val < soa->pbest_fun[s]) {
                soa->pbest_fun[s] = temp_val;
                for (int j = 0; j < opt->dim; j++) {
                    soa->pbest_s[s * opt->dim + j] = opt->population[s].position[j];
                }
            }

            if (!opt->best_solution.position) {
                fprintf(stderr, "SOA_optimize: NULL best_solution.position\n");
                free(temp_population);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            if (temp_val < opt->best_solution.fitness) {
                opt->best_solution.fitness = temp_val;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[s].position[j];
                }
                fprintf(stderr, "SOA_optimize: New global best fitness=%f\n", temp_val);
            }
        }
        free(temp_population);

        // Shuffle population if no improvement
        if (error_prev <= opt->best_solution.fitness || gens % 1 == 0) {
            fprintf(stderr, "SOA_optimize: Shuffling population\n");
            int *perm = (int*)calloc(opt->population_size, sizeof(int));
            if (!perm) {
                fprintf(stderr, "SOA_optimize: Memory allocation failed for perm\n");
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }
            for (int i = 0; i < opt->population_size; i++) perm[i] = i;
            for (int i = opt->population_size - 1; i > 0; i--) {
                int j = rand() % (i + 1);
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
                if (temp_pop) free(temp_pop);
                if (temp_fit) free(temp_fit);
                if (temp_pbest_s) free(temp_pbest_s);
                if (temp_pbest_fun) free(temp_pbest_fun);
                if (temp_e_t_1) free(temp_e_t_1);
                if (temp_e_t_2) free(temp_e_t_2);
                if (temp_f_t_1) free(temp_f_t_1);
                if (temp_f_t_2) free(temp_f_t_2);
                if (soa->magic != SOA_MAGIC) SOA_free(soa);
                return;
            }

            for (int i = 0; i < opt->population_size; i++) {
                int idx = perm[i];
                if (!opt->population[idx].position) {
                    fprintf(stderr, "SOA_optimize: NULL population[%d].position during shuffle\n", idx);
                    free(temp_pop);
                    free(temp_fit);
                    free(temp_pbest_s);
                    free(temp_pbest_fun);
                    free(temp_e_t_1);
                    free(temp_e_t_2);
                    free(temp_f_t_1);
                    free(temp_f_t_2);
                    free(perm);
                    if (soa->magic != SOA_MAGIC) SOA_free(soa);
                    return;
                }
                for (int j = 0; j < opt->dim; j++) {
                    temp_pop[i * opt->dim + j] = opt->population[idx].position[j];
                    temp_pbest_s[i * opt->dim + j] = soa->pbest_s[idx * opt->dim + j];
                    temp_e_t_1[i * opt->dim + j] = soa->e_t_1[idx * opt->dim + j];
                    temp_e_t_2[i * opt->dim + j] = soa->e_t_2[idx * opt->dim + j];
                }
                temp_fit[i] = opt->population[idx].fitness;
                temp_pbest_fun[i] = soa->pbest_fun[idx];
                temp_f_t_1[i] = soa->f_t_1[idx];
                temp_f_t_2[i] = soa->f_t_2[idx];
            }

            for (int i = 0; i < opt->population_size; i++) {
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[i].position[j] = temp_pop[i * opt->dim + j];
                    soa->pbest_s[i * opt->dim + j] = temp_pbest_s[i * opt->dim + j];
                    soa->e_t_1[i * opt->dim + j] = temp_e_t_1[i * opt->dim + j];
                    soa->e_t_2[i * opt->dim + j] = temp_e_t_2[i * opt->dim + j];
                }
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
        }

        error_prev = opt->best_solution.fitness;
        update_region_bests(soa);
        enforce_bound_constraints(opt);
    }

    fprintf(stderr, "SOA_optimize: Optimization completed\n");
    if (soa->magic != SOA_MAGIC) {
        fprintf(stderr, "SOA_optimize: Freeing temporary SOA at %p\n", soa);
        SOA_free(soa);
    }
}
