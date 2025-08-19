#include "SFL.h"
#include <time.h>

// ⚙️ Generate a random double between min and max
double rand_double(double min, double max);

// ⚙️ Swap two integers
void swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// ⚙️ Partition for quicksort
int partition(Solution *population, int *indices, int low, int high) {
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
void quicksort_fitness(Solution *population, int *indices, int low, int high) {
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
    
    Solution *temp_pop = (Solution *)malloc(opt->population_size * sizeof(Solution));
    if (!temp_pop) {
        free(indices);
        fprintf(stderr, "Memory allocation failed in sort_population\n");
        return;
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        temp_pop[i].position = (double *)malloc(opt->dim * sizeof(double));
        if (!temp_pop[i].position) {
            for (int j = 0; j < i; j++) free(temp_pop[j].position);
            free(temp_pop);
            free(indices);
            fprintf(stderr, "Memory allocation failed in sort_population\n");
            return;
        }
        temp_pop[i].fitness = opt->population[indices[i]].fitness;
        for (int j = 0; j < opt->dim; j++) {
            temp_pop[i].position[j] = opt->population[indices[i]].position[j];
        }
    }
    
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = temp_pop[i].fitness;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = temp_pop[i].position[j];
        }
        free(temp_pop[i].position);
    }
    
    free(temp_pop);
    free(indices);
}

// 🌊 Check if a position is within bounds
int is_in_range(double *position, double *bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] < bounds[2 * i] || position[i] > bounds[2 * i + 1]) {
            return 0;
        }
    }
    return 1;
}

// 🌊 Run Frog Leaping Algorithm on a memeplex
void run_fla(Optimizer *opt, Solution *memeplex, int memeplex_size, SFLContext *ctx) {
    if (!opt || !memeplex || !ctx || memeplex_size <= 0) {
        fprintf(stderr, "Invalid parameters in run_fla\n");
        return;
    }
    
    int num_parents = (int)(NUM_PARENTS_RATIO * memeplex_size);
    if (num_parents < 2) num_parents = 2;
    
    // Calculate selection probabilities
    double *P = (double *)malloc(memeplex_size * sizeof(double));
    if (!P) {
        fprintf(stderr, "Memory allocation failed in run_fla\n");
        return;
    }
    double sum_P = 0.0;
    for (int i = 0; i < memeplex_size; i++) {
        P[i] = 2.0 * (memeplex_size + 1 - (i + 1)) / (memeplex_size * (memeplex_size + 1));
        sum_P += P[i];
    }
    for (int i = 0; i < memeplex_size; i++) {
        P[i] /= sum_P; // Normalize
    }
    
    // Calculate memeplex bounds
    double *lower_bound = (double *)malloc(opt->dim * sizeof(double));
    double *upper_bound = (double *)malloc(opt->dim * sizeof(double));
    if (!lower_bound || !upper_bound) {
        free(P);
        if (lower_bound) free(lower_bound);
        if (upper_bound) free(upper_bound);
        fprintf(stderr, "Memory allocation failed in run_fla\n");
        return;
    }
    for (int j = 0; j < opt->dim; j++) {
        lower_bound[j] = memeplex[0].position[j];
        upper_bound[j] = memeplex[0].position[j];
    }
    for (int i = 1; i < memeplex_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (memeplex[i].position[j] < lower_bound[j]) lower_bound[j] = memeplex[i].position[j];
            if (memeplex[i].position[j] > upper_bound[j]) upper_bound[j] = memeplex[i].position[j];
        }
    }
    
    for (int iter = 0; iter < MAX_FLA_ITER; iter++) {
        // Select parents
        int *parent_indices = (int *)malloc(num_parents * sizeof(int));
        if (!parent_indices) {
            free(P);
            free(lower_bound);
            free(upper_bound);
            fprintf(stderr, "Memory allocation failed in run_fla\n");
            return;
        }
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
        
        // Create subcomplex
        Solution *subcomplex = (Solution *)malloc(num_parents * sizeof(Solution));
        if (!subcomplex) {
            free(parent_indices);
            free(P);
            free(lower_bound);
            free(upper_bound);
            fprintf(stderr, "Memory allocation failed in run_fla\n");
            return;
        }
        for (int i = 0; i < num_parents; i++) {
            subcomplex[i].position = (double *)malloc(opt->dim * sizeof(double));
            if (!subcomplex[i].position) {
                for (int j = 0; j < i; j++) free(subcomplex[j].position);
                free(subcomplex);
                free(parent_indices);
                free(P);
                free(lower_bound);
                free(upper_bound);
                fprintf(stderr, "Memory allocation failed in run_fla\n");
                return;
            }
            subcomplex[i].fitness = memeplex[parent_indices[i]].fitness;
            for (int j = 0; j < opt->dim; j++) {
                subcomplex[i].position[j] = memeplex[parent_indices[i]].position[j];
            }
        }
        
        // Generate offsprings
        for (int k = 0; k < NUM_OFFSPRINGS; k++) {
            // Sort subcomplex
            int *sub_indices = (int *)malloc(num_parents * sizeof(int));
            if (!sub_indices) {
                for (int i = 0; i < num_parents; i++) free(subcomplex[i].position);
                free(subcomplex);
                free(parent_indices);
                free(P);
                free(lower_bound);
                free(upper_bound);
                fprintf(stderr, "Memory allocation failed in run_fla\n");
                return;
            }
            for (int i = 0; i < num_parents; i++) sub_indices[i] = i;
            quicksort_fitness(subcomplex, sub_indices, 0, num_parents - 1);
            
            // Improvement Step 1: Move worst towards best in subcomplex
            double *new_pos = (double *)malloc(opt->dim * sizeof(double));
            if (!new_pos) {
                free(sub_indices);
                for (int i = 0; i < num_parents; i++) free(subcomplex[i].position);
                free(subcomplex);
                free(parent_indices);
                free(P);
                free(lower_bound);
                free(upper_bound);
                fprintf(stderr, "Memory allocation failed in run_fla\n");
                return;
            }
            int worst_idx = sub_indices[num_parents - 1];
            int best_idx = sub_indices[0];
            int improvement_step2 = 0;
            int censorship = 0;
            
            for (int j = 0; j < opt->dim; j++) {
                double step = SFL_STEP_SIZE * rand_double(0.0, 1.0) * (subcomplex[best_idx].position[j] - subcomplex[worst_idx].position[j]);
                new_pos[j] = subcomplex[worst_idx].position[j] + step;
            }
            
            if (is_in_range(new_pos, opt->bounds, opt->dim)) {
                double new_fitness = ctx->objective_function(new_pos);
                if (new_fitness < subcomplex[worst_idx].fitness) {
                    subcomplex[worst_idx].fitness = new_fitness;
                    for (int j = 0; j < opt->dim; j++) {
                        subcomplex[worst_idx].position[j] = new_pos[j];
                    }
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
                        for (int j = 0; j < opt->dim; j++) {
                            subcomplex[worst_idx].position[j] = new_pos[j];
                        }
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
            
            free(new_pos);
            
            // Update memeplex
            for (int i = 0; i < num_parents; i++) {
                memeplex[parent_indices[i]].fitness = subcomplex[sub_indices[i]].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    memeplex[parent_indices[i]].position[j] = subcomplex[sub_indices[i]].position[j];
                }
            }
            
            free(sub_indices);
        }
        
        for (int i = 0; i < num_parents; i++) {
            free(subcomplex[i].position);
        }
        free(subcomplex);
        free(parent_indices);
    }
    
    free(P);
    free(lower_bound);
    free(upper_bound);
}

// 🚀 Main Optimization Function
void SFL_optimize(void *opt_ptr, double (*objective_function)(double *)) {
    Optimizer *opt = (Optimizer *)opt_ptr;
    if (!opt || !objective_function || opt->population_size <= 0 || opt->dim <= 0 || opt->max_iter <= 0) {
        fprintf(stderr, "Invalid optimizer parameters\n");
        return;
    }
    
    srand(time(NULL)); // Seed random number generator
    
    // Set optimize function pointer
    opt->optimize = SFL_optimize;
    
    // Initialize SFL context
    SFLContext ctx;
    ctx.objective_function = objective_function;
    
    // Initialize population
    initialize_population_sfl(opt);
    
    // Sort population and set initial best solution
    sort_population(opt, &ctx);
    opt->best_solution.fitness = opt->population[0].fitness;
    for (int j = 0; j < opt->dim; j++) {
        opt->best_solution.position[j] = opt->population[0].position[j];
    }
    
    // Create memeplex indices
    int **memeplex_indices = (int **)malloc(NUM_MEMEPLEXES * sizeof(int *));
    if (!memeplex_indices) {
        fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
        return;
    }
    for (int i = 0; i < NUM_MEMEPLEXES; i++) {
        memeplex_indices[i] = (int *)malloc(MEMEPLEX_SIZE * sizeof(int));
        if (!memeplex_indices[i]) {
            for (int j = 0; j < i; j++) free(memeplex_indices[j]);
            free(memeplex_indices);
            fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
            return;
        }
        for (int j = 0; j < MEMEPLEX_SIZE; j++) {
            memeplex_indices[i][j] = i + j * NUM_MEMEPLEXES; // Distribute frogs
        }
    }
    
    // Main optimization loop
    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int m = 0; m < NUM_MEMEPLEXES; m++) {
            // Create memeplex
            Solution *memeplex = (Solution *)malloc(MEMEPLEX_SIZE * sizeof(Solution));
            if (!memeplex) {
                for (int i = 0; i < NUM_MEMEPLEXES; i++) free(memeplex_indices[i]);
                free(memeplex_indices);
                fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
                return;
            }
            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                memeplex[i].position = (double *)malloc(opt->dim * sizeof(double));
                if (!memeplex[i].position) {
                    for (int j = 0; j < i; j++) free(memeplex[j].position);
                    free(memeplex);
                    for (int j = 0; j < NUM_MEMEPLEXES; j++) free(memeplex_indices[j]);
                    free(memeplex_indices);
                    fprintf(stderr, "Memory allocation failed in SFL_optimize\n");
                    return;
                }
                memeplex[i].fitness = opt->population[memeplex_indices[m][i]].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    memeplex[i].position[j] = opt->population[memeplex_indices[m][i]].position[j];
                }
            }
            
            // Run FLA
            run_fla(opt, memeplex, MEMEPLEX_SIZE, &ctx);
            
            // Update population
            for (int i = 0; i < MEMEPLEX_SIZE; i++) {
                opt->population[memeplex_indices[m][i]].fitness = memeplex[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->population[memeplex_indices[m][i]].position[j] = memeplex[i].position[j];
                }
                free(memeplex[i].position);
            }
            free(memeplex);
        }
        
        // Update fitness
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = ctx.objective_function(opt->population[i].position);
        }
        
        // Sort population
        sort_population(opt, &ctx);
        
        // Update best solution
        if (opt->population[0].fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = opt->population[0].fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[0].position[j];
            }
        }
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    // Cleanup
    for (int i = 0; i < NUM_MEMEPLEXES; i++) {
        free(memeplex_indices[i]);
    }
    free(memeplex_indices);
}
