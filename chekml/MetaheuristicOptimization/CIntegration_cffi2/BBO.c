#include "BBO.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize habitats randomly within bounds
void bbo_initialize_habitats(Optimizer *opt, BBOData *data) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        opt->population[i].fitness = INFINITY;
    }
    enforce_bound_constraints(opt);
}

// Compute migration rates (mu and lambda)
void compute_migration_rates(double *mu, double *lambda_, int population_size) {
    for (int i = 0; i < population_size; i++) {
        mu[i] = 1.0 - ((double)i / (population_size - 1));
        lambda_[i] = 1.0 - mu[i];
    }
}

// Roulette wheel selection
int roulette_wheel_selection_bbo(double *probabilities, int size) {
    double r = rand_double(0.0, 1.0);
    double cumsum = 0.0;
    for (int i = 0; i < size; i++) {
        cumsum += probabilities[i];
        if (r <= cumsum) return i;
    }
    return size - 1;
}

// Migration phase
void bbo_migration_phase(Optimizer *opt, BBOData *data) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (rand_double(0.0, 1.0) <= data->lambda_[i]) {
                // Compute emigration probabilities
                double *ep = (double *)malloc(opt->population_size * sizeof(double));
                double ep_sum = 0.0;
                for (int k = 0; k < opt->population_size; k++) {
                    ep[k] = (k == i) ? 0.0 : data->mu[k];
                    ep_sum += ep[k];
                }
                if (ep_sum > 0) {
                    for (int k = 0; k < opt->population_size; k++) {
                        ep[k] /= ep_sum;
                    }
                    int source_idx = roulette_wheel_selection_bbo(ep, opt->population_size);
                    opt->population[i].position[j] += BBO_ALPHA * (opt->population[source_idx].position[j] - opt->population[i].position[j]);
                }
                free(ep);
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Mutation phase
void bbo_mutation_phase(Optimizer *opt, BBOData *data) {
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (rand_double(0.0, 1.0) <= MUTATION_PROB) {
                // Approximate normal distribution using Box-Muller transform
                double u1 = rand_double(0.0, 1.0);
                double u2 = rand_double(0.0, 1.0);
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                opt->population[i].position[j] += data->mutation_sigma * z;
            }
        }
    }
    enforce_bound_constraints(opt);
}

// Selection phase (keep best and replace worst)
void bbo_selection_phase(Optimizer *opt, BBOData *data) {
    // Create a temporary array for sorting
    typedef struct {
        double *position;
        double fitness;
        int original_idx;
    } Habitat;
    
    Habitat *habitats = (Habitat *)malloc(opt->population_size * sizeof(Habitat));
    for (int i = 0; i < opt->population_size; i++) {
        habitats[i].position = opt->population[i].position;
        habitats[i].fitness = opt->population[i].fitness;
        habitats[i].original_idx = i;
    }
    
    // Sort habitats by fitness
    for (int i = 0; i < opt->population_size - 1; i++) {
        for (int j = 0; j < opt->population_size - i - 1; j++) {
            if (habitats[j].fitness > habitats[j + 1].fitness) {
                Habitat temp = habitats[j];
                habitats[j] = habitats[j + 1];
                habitats[j + 1] = temp;
            }
        }
    }
    
    // Keep best n_keep and replace others
    int n_keep = (int)(KEEP_RATE * opt->population_size);
    int n_new = opt->population_size - n_keep;
    
    // Copy back the best n_keep
    for (int i = 0; i < n_keep; i++) {
        int idx = habitats[i].original_idx;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->population[idx].position[j];
        }
        opt->population[i].fitness = habitats[i].fitness;
    }
    
    // Copy the next n_new from sorted habitats
    for (int i = 0; i < n_new; i++) {
        int idx = habitats[i].original_idx;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[n_keep + i].position[j] = opt->population[idx].position[j];
        }
        opt->population[n_keep + i].fitness = habitats[i].fitness;
    }
    
    free(habitats);
    enforce_bound_constraints(opt);
}

// Main Optimization Function
void BBO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    // Initialize BBO-specific data
    BBOData *data = (BBOData *)malloc(sizeof(BBOData));
    data->mu = (double *)malloc(opt->population_size * sizeof(double));
    data->lambda_ = (double *)malloc(opt->population_size * sizeof(double));
    data->mutation_sigma = MUTATION_SCALE * (opt->bounds[1] - opt->bounds[0]);
    data->history = (void *)malloc(opt->max_iter * sizeof(*data->history));
    for (int i = 0; i < opt->max_iter; i++) {
        data->history[i].solution = (double *)malloc(opt->dim * sizeof(double));
    }
    
    compute_migration_rates(data->mu, data->lambda_, opt->population_size);
    
    bbo_initialize_habitats(opt, data);
    
    for (int iter = 0; iter < opt->max_iter; iter++) {
        // Evaluate fitness
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        
        bbo_migration_phase(opt, data);
        bbo_mutation_phase(opt, data);
        
        // Re-evaluate after migration and mutation
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].fitness = objective_function(opt->population[i].position);
            if (opt->population[i].fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = opt->population[i].fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        
        bbo_selection_phase(opt, data);
        
        // Store history in BBOData
        data->history[iter].iteration = iter;
        data->history[iter].value = opt->best_solution.fitness;
        for (int j = 0; j < opt->dim; j++) {
            data->history[iter].solution[j] = opt->best_solution.position[j];
        }
        
        printf("Iteration %d: Best Value = %f\n", iter + 1, opt->best_solution.fitness);
    }
    
    // Clean up
    for (int i = 0; i < opt->max_iter; i++) {
        free(data->history[i].solution);
    }
    free(data->history);
    free(data->mu);
    free(data->lambda_);
    free(data);
}
