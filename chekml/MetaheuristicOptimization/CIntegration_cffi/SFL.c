#include "SFL.h"
#include <time.h>
#include <string.h>

// ⚙️ Generate a random double between min and max
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// ⚙️ Swap two integers
static inline void swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// ⚙️ Partition for quicksort
static int partition(Solution *population, int *indices, int low, int high) {
    double pivot = population[indices[high]].fitness;
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (population[indices[j]].fitness <= pivot) {
            i++;
            swap_int(&indices[i], &indices[j]);
        }
    }
    swap_int(&indices[i + 1], &indices[high]);
    return i + 1;
}

// ⚙️ Quicksort for sorting population by fitness
static void quicksort_fitness(Solution *population, int *indices, int low, int high) {
    if (low < high) {
        int pi = partition(population, indices, low, high);
        quicksort_fitness(population, indices, low, pi - 1);
        quicksort_fitness(population, indices, pi + 1, high);
    }
}

// 🌊 Initialize population with random positions
void initialize_population_sfl(Optimizer *opt) {
    if (!opt || opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Invalid optimizer parameters in initialize_population_sfl\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// 🌊 Sort population by fitness
void sort_population(Optimizer *opt, SFLContext *ctx) {
    if (!opt || !ctx || opt->population_size <= 0) {
        fprintf(stderr, "Invalid parameters in sort_population\n");
        return;
    }

    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed in sort_population\n");
        return;
    }

    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
        if (opt->population[i].fitness == INFINITY) {
            opt->population[i].fitness = ctx->objective_function(opt->population[i].position);
        }
    }

    quicksort_fitness(opt->population, indices, 0, opt->population_size - 1);

    // Reuse population array to avoid extra allocation
    Solution *temp = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!temp) {
        free(indices);
        fprintf(stderr, "Memory allocation failed in sort_population\n");
        return;
    }
    memcpy(temp, opt->population, opt->population_size * sizeof(Solution));

    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = temp[indices[i]].fitness;
        memcpy(opt->population[i].position, temp[indices[i]].position, opt->dim * sizeof(double));
    }

    free(temp);
    free(indices);
}

// 🌊 Check if a position is within bounds
int is_in_range(const double *position, const double *bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] < bounds[2 * i] || position[i] > bounds[2 * i + 1]) {
            return 0;
        }
    }
    return 1;
}

// 🌊 Run Frog Leaping Algorithm on a memeplex
void run_fla(Optimizer *opt, Solution *memeplex, int memeplex_size, SFLContext *ctx, double *P) {
    if (!opt || !memeplex || !ctx || memeplex_size <= 0) {
        fprintf(stderr, "Invalid parameters in run_fla\n");
        return;
    }

    int num_parents = (int)(NUM_PARENTS_RATIO * memeplex_size);
    if (num_parents < 2) num_parents = 2;

    // Preallocate reusable buffers
    double *lower_bound = (double *)malloc(opt->dim * sizeof(double));
    double *upper_bound = (double *)malloc(opt->dim * sizeof(double));
    int *parent_indices = (int *)malloc(num_parents * sizeof(int));
    Solution *subcomplex = (Solution *)malloc(num_parents * sizeof(Solution));
    int *sub_indices = (int *)malloc(num_parents * sizeof(int));
    double *new_pos = (double *)malloc(opt->dim * sizeof(double));

    if (!lower_bound || !upper_bound || !parent_indices || !subcomplex || !sub_indices || !new_pos) {
        free(lower_bound); free(upper_bound); free(parent_indices);
        free(subcomplex); free(sub_indices); free(new_pos);
        fprintf(stderr, "Memory allocation failed in run_fla\n");
        return;
    }

    // Initialize subcomplex positions
    for (int i = 0; i < num_parents; i++) {
        subcomplex[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!subcomplex[i].position) {
            for (int j = 0; j < i; j++) free(subcomplex[j].position);
            free(lower_bound); free(upper_bound); free(parent_indices);
            free(subcomplex); free(sub_indices); free(new_pos);
            fprintf(stderr, "Memory allocation failed in run_fla\n");
            return;
        }
    }

    // Calculate memeplex bounds once
    memcpy(lower_bound, memeplex[0].position, opt->dim * sizeof(double));
    memcpy(upper_bound, memeplex[0].position, opt->dim * sizeof(double));
    for (int i = 1; i < memeplex_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (memeplex[i].position[j] < lower_bound[j]) lower_bound[j] = memeplex[i].position[j];
            if (memeplex[i].position[j] > upper_bound[j]) upper_bound[j] = memeplex[i].position[j];
        }
    }

    for (int iter = 0; iter < MAX_FLA_ITER; iter++) {
        // Select parents using precomputed probabilities
        for (int i = 0; i < num_parents; i++) {
            double r = rand_double(0.0, 1.0);
            double cumsum = 0.0;
            for (int j = 0; j < memeplex_size; j++) {
                cumsum += P[j];
                if (r <= cumsum) {
                    parent_indices[i] = j;
                    break;
                }
            }
        }

        // Populate subcomplex
        for (int i = 0; i < num_parents; i++) {
            subcomplex[i].fitness = memeplex[parent_indices[i]].fitness;
            memcpy(subcomplex[i].position, memeplex[parent_indices[i]].position, opt->dim * sizeof(double));
        }

        // Generate offsprings
        for (int k = 0; k < NUM_OFFSPRINGS; k++) {
            // Sort subcomplex
            for (int i = 0; i < num_parents; i++) sub_indices[i] = i;
            quicksort_fitness(subcomplex, sub_indices, 0, num_parents - 1);

            int worst_idx = sub_indices[num_parents - 1];
            int best_idx = sub_indices[0];
            int improvement_step2 = 0;
            int censorship = 0;

            // Improvement Step 1: Move worst towards best in subcomplex
            for (int j = 0; j < opt->dim; j++) {
                double step = SFL_STEP_SIZE * rand_double(0.0, 1.0) * (subcomplex[best_idx].position[j] - subcomplex[worst_idx].position[j]);
                new_pos[j] = subcomplex[worst_idx].position[j] + step;
            }

            if (is_in_range(new_pos, opt->bounds, opt->dim)) {
                double new_fitness = ctx->objective_function(new_pos);
                if (new_fitness < subcomplex[worst_idx].fitness) {
                    subcomplex[worst_idx].fitness = new_fitness;
                    memcpy(subcomplex[worst_idx].position, new_pos, opt->dim * sizeof(double));
                } else {
                    improvement_step2 = 1;
                }
            } else {
                improvement_step2 = 1;
            }

            // Improvement Step 2: Move worst towards global best
            if (improvement_step2) {
                for (int j = 0; j < opt->dim; j++) {
                    double step = SFL_STEP_SIZE * rand_double(0.0, 1.0) * (opt->best_solution.position[j] - subcomplex[worst_idx].position[j]);
                    new_pos[j] = subcomplex[worst_idx].position[j] + step;
                }
                if (is_in_range(new_pos, opt->bounds, opt->dim)) {
                    double new_fitness = ctx->objective_function(new_pos);
                    if (new_fitness < subcomplex[worst_idx].fitness) {
                        subcomplex[worst_idx].fitness = new_fitness;
                        memcpy(subcomplex[worst_idx].position, new_pos, opt->dim * sizeof(double));
                    } else {
                        censorship = 1;
                    }
                } else {
                    censorship = 1;
                }
            }

            // Censorship
            if (censorship) {
                for (int j = 0; j < opt->dim; j++) {
                    subcomplex[worst_idx].position[j] = rand_double(lower_bound[j], upper_bound[j]);
                }
                subcomplex[worst_idx].fitness = ctx->objective_function(subcomplex[worst_idx].position);
            }

            // Update memeplex
            for (int i = 0; i < num_parents; i++) {
                memeplex[parent_indices[i]].fitness = subcomplex[i].fitness;
                memcpy(memeplex[parent_indices[i]].position, subcomplex[i].position, opt->dim * sizeof(double));
            }
        }
    }

    // Cleanup
    for (int i = 0; i < num_parents; i++) free(subcomplex[i].position);
    free(lower_bound); free(upper_bound); free(parent_indices);
    free(subcomplex); free(sub_indices); free(new_pos);
}

// 🚀 Main Optimization Function
void SFL_optimize(void *opt_ptr, double (*objective_function)(double *)) {
    Optimizer *opt = (Optimizer *)opt_ptr;
    if (!opt || !objective_function || opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Invalid optimizer parameters\n");
        return;
    }

    srand(time(NULL));

    opt->optimize = SFL_optimize;
    SFLContext ctx = { .objective_function = objective_function };

    // Precompute selection probabilities
    double *P = (double *)malloc(MEMEPLEX_SIZE * sizeof(double));
    if (!P) {
        fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
        return;
    }
    double sum_P = 0.0;
    for (int i = 0; i < MEMEPLEX_SIZE; i++) {
        P[i] = 2.0 * (MEMEPLEX_SIZE + 1 - (i + 1)) / (MEMEPLEX_SIZE * (MEMEPLEX_SIZE + 1));
        sum_P += P[i];
    }
    for (int i = 0; i < MEMEPLEX_SIZE; i++) {
        P[i] /= sum_P;
    }

    // Initialize population
    initialize_population_sfl(opt);
    sort_population(opt, &ctx);
    opt->best_solution.fitness = opt->population[0].fitness;
    memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));

    // Preallocate memeplex indices
    int **memeplex_indices = (int **)malloc(NUM_MEMEPLEXES * sizeof(int *));
    Solution *memeplex = (Solution *)malloc(MEMEPLEX_SIZE * sizeof(Solution));
    if (!memeplex_indices || !memeplex) {
        free(P); free(memeplex_indices); free(memeplex);
        fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
        return;
    }
    for (int i = 0; i < NUM_MEMEPLEXES; i++) {
        memeplex_indices[i] = (int *)malloc(MEMEPLEX_SIZE * sizeof(int));
        if (!memeplex_indices[i]) {
            for (int j = 0; j < i; j++) free(memeplex_indices[j]);
            free(P); free(memeplex_indices); free(memeplex);
            fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
            return;
        }
        for (int j = 0; j < MEMEPLEX_SIZE; j++) {
            memeplex_indices[i][j] = i + j * NUM_MEMEPLEXES;
        }
    }
    for (int i = 0; i < MEMEPLEX_SIZE; i++) {
        memeplex[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!memeplex[i].position) {
            for (int j = 0; j < i; j++) free(memeplex[j].position);
            for (int j = 0; j < NUM_MEMEPLEXES; j++) free(memeplex_indices[j]);
            free(P); free(memeplex_indices); free(memeplex);
            fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
            return;
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int m = 0; m < NUM_MEMEPLEXES; m++) {
            // Populate memeplex
            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                memeplex[i].fitness = opt->population[memeplex_indices[m][i]].fitness;
                memcpy(memeplex[i].position, opt->population[memeplex_indices[m][i]].position, opt->dim * sizeof(double));
            }

            // Run FLA
            run_fla(opt, memeplex, MEMEPLEX_SIZE, &ctx, P);

            // Update population
            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                opt->population[memeplex_indices[m][i]].fitness = memeplex[i].fitness;
                memcpy(opt->population[memeplex_indices[m][i]].position, memeplex[i].position, opt->dim * sizeof(double));
            }
        }

        // Sort population and update best solution
        sort_population(opt, &ctx);
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            memcpy(opt->best_solution.position, opt->population[0].position, opt->dim * sizeof(double));
        }

        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }

    // Cleanup
    for (int i = 0; i < MEMEPLEX_SIZE; i++) free(memeplex[i].position);
    for (int i = 0; i < NUM_MEMEPLEXES; i++) free(memeplex_indices[i]);
    free(P); free(memeplex_indices); free(memeplex);
}
