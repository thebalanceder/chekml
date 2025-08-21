#include "MFO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Constants
#define TWO_PI 6.283185307179586  // 2 * M_PI
#define LOG_E 0.4342944819032518  // 1 / ln(10) for exponential decay

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Create MFOData structure with pre-allocated buffers
MFOData* mfo_create_data(int population_size, int dim) {
    MFOData *mfo_data = (MFOData *)malloc(sizeof(MFOData));
    mfo_data->objective_function = NULL;
    mfo_data->iteration = 0;
    
    // Persistent arrays
    mfo_data->previous_population = (Solution *)malloc(population_size * sizeof(Solution));
    mfo_data->previous_fitness = (double *)malloc(population_size * sizeof(double));
    mfo_data->best_flames = (Solution *)malloc(population_size * sizeof(Solution));
    mfo_data->best_flame_fitness = (double *)malloc(population_size * sizeof(double));

    // Pre-allocated temporary buffers
    mfo_data->sorted_population = (Solution *)malloc(population_size * sizeof(Solution));
    mfo_data->sorted_fitness = (double *)malloc(population_size * sizeof(double));
    mfo_data->indices = (int *)malloc(population_size * sizeof(int));
    mfo_data->temp_population = (Solution *)malloc(2 * population_size * sizeof(Solution));
    mfo_data->temp_fitness = (double *)malloc(2 * population_size * sizeof(double));
    mfo_data->temp_indices = (int *)malloc(2 * population_size * sizeof(int));

    for (int i = 0; i < population_size; i++) {
        mfo_data->previous_population[i].position = (double *)malloc(dim * sizeof(double));
        mfo_data->best_flames[i].position = (double *)malloc(dim * sizeof(double));
        mfo_data->sorted_population[i].position = (double *)malloc(dim * sizeof(double));
        mfo_data->previous_fitness[i] = INFINITY;
        mfo_data->best_flame_fitness[i] = INFINITY;
    }
    for (int i = 0; i < 2 * population_size; i++) {
        mfo_data->temp_population[i].position = (double *)malloc(dim * sizeof(double));
    }

    return mfo_data;
}

// Free MFOData structure
void mfo_free_data(MFOData *mfo_data, int population_size) {
    for (int i = 0; i < population_size; i++) {
        free(mfo_data->previous_population[i].position);
        free(mfo_data->best_flames[i].position);
        free(mfo_data->sorted_population[i].position);
    }
    for (int i = 0; i < 2 * population_size; i++) {
        free(mfo_data->temp_population[i].position);
    }
    free(mfo_data->previous_population);
    free(mfo_data->previous_fitness);
    free(mfo_data->best_flames);
    free(mfo_data->best_flame_fitness);
    free(mfo_data->sorted_population);
    free(mfo_data->sorted_fitness);
    free(mfo_data->indices);
    free(mfo_data->temp_population);
    free(mfo_data->temp_fitness);
    free(mfo_data->temp_indices);
    free(mfo_data);
}

// Comparison function for qsort
int compare_fitness_mfo(const void *a, const void *b) {
    double fitness_a = ((Solution *)a)->fitness;
    double fitness_b = ((Solution *)b)->fitness;
    return (fitness_a > fitness_b) - (fitness_a < fitness_b);
}

// Sort population by fitness and store in sorted arrays
void mfo_sort_population(Optimizer *opt, MFOData *mfo_data) {
    // Copy population to temporary array
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(mfo_data->sorted_population[i].position, opt->population[i].position, opt->dim * sizeof(double));
        mfo_data->sorted_population[i].fitness = opt->population[i].fitness;
        mfo_data->indices[i] = i;
    }

    // Sort using qsort
    qsort(mfo_data->sorted_population, opt->population_size, sizeof(Solution), compare_fitness_mfo);

    // Update sorted fitness
    for (int i = 0; i < opt->population_size; i++) {
        mfo_data->sorted_fitness[i] = mfo_data->sorted_population[i].fitness;
    }
}

// Update flames (combine previous population and best flames)
void mfo_update_flames(Optimizer *opt, MFOData *mfo_data) {
    // Compute fitness for current population
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = mfo_data->objective_function(opt->population[i].position);
    }

    if (mfo_data->iteration == 0) {
        // Sort initial population
        mfo_sort_population(opt, mfo_data);

        // Initialize best flames
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(mfo_data->best_flames[i].position, mfo_data->sorted_population[i].position, opt->dim * sizeof(double));
            mfo_data->best_flame_fitness[i] = mfo_data->sorted_fitness[i];
        }
    } else {
        // Combine previous population and best flames
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(mfo_data->temp_population[i].position, mfo_data->previous_population[i].position, opt->dim * sizeof(double));
            mfo_data->temp_population[i].fitness = mfo_data->previous_fitness[i];
            memcpy(mfo_data->temp_population[i + opt->population_size].position, mfo_data->best_flames[i].position, opt->dim * sizeof(double));
            mfo_data->temp_population[i + opt->population_size].fitness = mfo_data->best_flame_fitness[i];
        }

        // Sort combined population
        qsort(mfo_data->temp_population, 2 * opt->population_size, sizeof(Solution), compare_fitness_mfo);

        // Update best flames
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            memcpy(mfo_data->best_flames[i].position, mfo_data->temp_population[i].position, opt->dim * sizeof(double));
            mfo_data->best_flame_fitness[i] = mfo_data->temp_population[i].fitness;
        }
    }

    // Update best solution
    if (mfo_data->sorted_fitness[0] < opt->best_solution.fitness) {
        opt->best_solution.fitness = mfo_data->sorted_fitness[0];
        memcpy(opt->best_solution.position, mfo_data->sorted_population[0].position, opt->dim * sizeof(double));
        printf("Updated best solution at iteration %d: fitness=%f, pos=[%f, %f]\n",
               mfo_data->iteration, opt->best_solution.fitness,
               opt->best_solution.position[0], opt->best_solution.position[1]);
    }

    // Store current population as previous
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        memcpy(mfo_data->previous_population[i].position, opt->population[i].position, opt->dim * sizeof(double));
        mfo_data->previous_fitness[i] = opt->population[i].fitness;
    }
}

// Update moth positions based on flames
void mfo_update_moth_positions(Optimizer *opt, MFOData *mfo_data) {
    // Calculate number of flames (Eq. 3.14)
    int flame_no = (int)round(opt->population_size - mfo_data->iteration * ((opt->population_size - 1.0) / opt->max_iter));

    // Non-linear decay for 'a' (faster convergence)
    double t = (double)mfo_data->iteration / opt->max_iter;
    double a = MFO_A_INITIAL - (MFO_A_INITIAL - MFO_A_FINAL) * (1.0 - exp(-3.0 * t));

    // Sort current population
    mfo_sort_population(opt, mfo_data);

    // Update moth positions
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            double distance_to_flame, t_rand, spiral;
            t_rand = (a - 1.0) * rand_double(0.0, 1.0) + 1.0;
            spiral = exp(MFO_B_CONSTANT * t_rand) * cos(t_rand * TWO_PI);
            
            if (i < flame_no) {
                // Update w.r.t. corresponding flame (Eq. 3.12)
                distance_to_flame = fabs(mfo_data->sorted_population[i].position[j] - opt->population[i].position[j]);
                opt->population[i].position[j] = distance_to_flame * spiral + mfo_data->sorted_population[i].position[j];
            } else {
                // Update w.r.t. best flame (Eq. 3.12)
                distance_to_flame = fabs(mfo_data->sorted_population[flame_no - 1].position[j] - opt->population[i].position[j]);
                opt->population[i].position[j] = distance_to_flame * spiral + mfo_data->sorted_population[flame_no - 1].position[j];
            }
        }
    }

    // Enforce bounds
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void MFO_optimize(void *opt_void, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)opt_void;
    MFOData *mfo_data = mfo_create_data(opt->population_size, opt->dim);
    mfo_data->objective_function = objective_function;
    mfo_data->iteration = 0;

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize fitness for population
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = mfo_data->objective_function(opt->population[i].position);
        if (opt->population[i].fitness < opt->best_solution.fitness) {
            #pragma omp critical
            {
                if (opt->population[i].fitness < opt->best_solution.fitness) {
                    opt->best_solution.fitness = opt->population[i].fitness;
                    memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                }
            }
        }
    }

    printf("MFO is optimizing your problem\n");
    printf("Initial best solution: pos=[%f, %f], fitness=%f\n",
           opt->best_solution.position[0], opt->best_solution.position[1],
           opt->best_solution.fitness);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        mfo_data->iteration = iter;

        // Update flames and best solution
        mfo_update_flames(opt, mfo_data);

        // Update moth positions
        mfo_update_moth_positions(opt, mfo_data);

        // Update fitness for all moths
        #pragma omp parallel for
        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = mfo_data->objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                #pragma omp critical
                {
                    if (new_fitness < opt->best_solution.fitness) {
                        opt->best_solution.fitness = new_fitness;
                        memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
                        printf("Updated best solution at iteration %d: fitness=%f, pos=[%f, %f]\n",
                               iter, opt->best_solution.fitness,
                               opt->best_solution.position[0], opt->best_solution.position[1]);
                    }
                }
            }
        }

        // Log progress every 100 iterations
        if (iter % 100 == 0) {
            printf("At iteration %d, the best fitness is %f\n", iter, opt->best_solution.fitness);
        }
    }

    // Clean up MFO data
    mfo_free_data(mfo_data, opt->population_size);
}
