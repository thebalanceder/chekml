#include "CheRO.h"
#include <stdlib.h>
#include <time.h>

// Generate random double between min and max
double rand_double(double min, double max);

// Initialize molecules with random positions and kinetic energies
void initialize_molecules(CROOptimizer *cro) {
    cro->molecules = (Molecule *)malloc(cro->population_size * sizeof(Molecule));
    for (int i = 0; i < cro->population_size; i++) {
        cro->molecules[i].position = (double *)malloc(cro->opt->dim * sizeof(double));
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[i].position[j] = rand_double(cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->molecules[i].ke = CRO_INITIAL_KE;
    }
    cro->buffer = CRO_BUFFER_INITIAL;
}

// Evaluate potential energy (fitness) for all molecules
void evaluate_molecules(CROOptimizer *cro) {
    for (int i = 0; i < cro->population_size; i++) {
        cro->molecules[i].pe = cro->objective_function(cro->molecules[i].position);
        if (cro->molecules[i].pe < cro->opt->best_solution.fitness) {
            cro->opt->best_solution.fitness = cro->molecules[i].pe;
            for (int j = 0; j < cro->opt->dim; j++) {
                cro->opt->best_solution.position[j] = cro->molecules[i].position[j];
            }
        }
    }
}

// On-wall ineffective collision (local search)
void on_wall_collision(CROOptimizer *cro, int index) {
    double *new_solution = (double *)malloc(cro->opt->dim * sizeof(double));
    double r = rand_double(0.0, 1.0);
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution[j] = cro->molecules[index].position[j] + 
                          r * (cro->opt->bounds[2 * j + 1] - cro->opt->bounds[2 * j]) * rand_double(-1.0, 1.0);
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe = cro->molecules[index].pe;
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe + cro->molecules[index].ke >= new_pe) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index].position[j] = new_solution[j];
        }
        cro->molecules[index].pe = new_pe;
        cro->molecules[index].ke = old_pe + cro->molecules[index].ke - new_pe;
    }
    free(new_solution);
}

// Decomposition reaction (global exploration)
void decomposition(CROOptimizer *cro, int index) {
    double old_pe = cro->molecules[index].pe;
    if (cro->molecules[index].ke + old_pe < CRO_ALPHA) {
        return;
    }
    
    double *split1 = (double *)malloc(cro->opt->dim * sizeof(double));
    double *split2 = (double *)malloc(cro->opt->dim * sizeof(double));
    for (int j = 0; j < cro->opt->dim; j++) {
        split1[j] = cro->molecules[index].position[j] + CRO_SPLIT_RATIO * rand_double(-1.0, 1.0);
        split2[j] = cro->molecules[index].position[j] - CRO_SPLIT_RATIO * rand_double(-1.0, 1.0);
        split1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split1[j]));
        split2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], split2[j]));
    }
    
    double pe1 = cro->objective_function(split1);
    double pe2 = cro->objective_function(split2);
    
    if (old_pe + cro->molecules[index].ke >= pe1 + pe2) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index].position[j] = split1[j];
        }
        cro->molecules[index].pe = pe1;
        cro->molecules[index].ke = (old_pe + cro->molecules[index].ke - pe1 - pe2) / 2.0;
        
        cro->molecules = (Molecule *)realloc(cro->molecules, (cro->population_size + 1) * sizeof(Molecule));
        cro->molecules[cro->population_size].position = split2;
        cro->molecules[cro->population_size].pe = pe2;
        cro->molecules[cro->population_size].ke = (old_pe + cro->molecules[index].ke - pe1 - pe2) / 2.0;
        cro->population_size++;
    } else {
        free(split2);
        free(split1);
    }
}

// Inter-molecular ineffective collision (local search)
void inter_molecular_collision(CROOptimizer *cro, int index1, int index2) {
    double *new_solution1 = (double *)malloc(cro->opt->dim * sizeof(double));
    double *new_solution2 = (double *)malloc(cro->opt->dim * sizeof(double));
    double r1 = rand_double(0.0, 1.0);
    double r2 = rand_double(0.0, 1.0);
    
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution1[j] = cro->molecules[index1].position[j] + r1 * rand_double(-1.0, 1.0);
        new_solution2[j] = cro->molecules[index2].position[j] + r2 * rand_double(-1.0, 1.0);
        new_solution1[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution1[j]));
        new_solution2[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution2[j]));
    }
    
    double old_pe1 = cro->molecules[index1].pe;
    double old_pe2 = cro->molecules[index2].pe;
    double new_pe1 = cro->objective_function(new_solution1);
    double new_pe2 = cro->objective_function(new_solution2);
    
    if (old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke >= new_pe1 + new_pe2) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index1].position[j] = new_solution1[j];
            cro->molecules[index2].position[j] = new_solution2[j];
        }
        cro->molecules[index1].pe = new_pe1;
        cro->molecules[index2].pe = new_pe2;
        double total_ke = old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke - new_pe1 - new_pe2;
        cro->molecules[index1].ke = total_ke * rand_double(0.0, 1.0);
        cro->molecules[index2].ke = total_ke - cro->molecules[index1].ke;
    }
    free(new_solution1);
    free(new_solution2);
}

// Synthesis reaction (global exploration)
void synthesis(CROOptimizer *cro, int index1, int index2) {
    double *new_solution = (double *)malloc(cro->opt->dim * sizeof(double));
    for (int j = 0; j < cro->opt->dim; j++) {
        new_solution[j] = (cro->molecules[index1].position[j] + cro->molecules[index2].position[j]) / 2.0;
        new_solution[j] = fmax(cro->opt->bounds[2 * j], fmin(cro->opt->bounds[2 * j + 1], new_solution[j]));
    }
    
    double old_pe1 = cro->molecules[index1].pe;
    double old_pe2 = cro->molecules[index2].pe;
    double new_pe = cro->objective_function(new_solution);
    
    if (old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke >= new_pe + CRO_BETA) {
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[index1].position[j] = new_solution[j];
        }
        cro->molecules[index1].pe = new_pe;
        cro->molecules[index1].ke = old_pe1 + old_pe2 + cro->molecules[index1].ke + cro->molecules[index2].ke - new_pe;
        cro->buffer += rand_double(0.0, 1.0) * cro->molecules[index1].ke;
        
        free(cro->molecules[index2].position);
        for (int i = index2; i < cro->population_size - 1; i++) {
            cro->molecules[i] = cro->molecules[i + 1];
        }
        cro->molecules = (Molecule *)realloc(cro->molecules, (cro->population_size - 1) * sizeof(Molecule));
        cro->population_size--;
    }
    free(new_solution);
}

// Elimination phase (replace worst molecules)
void elimination_phase_chero(CROOptimizer *cro) {
    int worst_count = (int)(CRO_ELIMINATION_RATIO * cro->population_size);
    double *fitness = (double *)malloc(cro->population_size * sizeof(double));
    int *indices = (int *)malloc(cro->population_size * sizeof(int));
    
    for (int i = 0; i < cro->population_size; i++) {
        fitness[i] = cro->molecules[i].pe;
        indices[i] = i;
    }
    
    // Simple bubble sort to find worst indices
    for (int i = 0; i < cro->population_size - 1; i++) {
        for (int j = 0; j < cro->population_size - i - 1; j++) {
            if (fitness[j] < fitness[j + 1]) {
                double temp_f = fitness[j];
                int temp_i = indices[j];
                fitness[j] = fitness[j + 1];
                indices[j] = indices[j + 1];
                fitness[j + 1] = temp_f;
                indices[j + 1] = temp_i;
            }
        }
    }
    
    for (int i = 0; i < worst_count; i++) {
        int idx = indices[i];
        for (int j = 0; j < cro->opt->dim; j++) {
            cro->molecules[idx].position[j] = rand_double(cro->opt->bounds[2 * j], cro->opt->bounds[2 * j + 1]);
        }
        cro->molecules[idx].ke = CRO_INITIAL_KE;
        cro->molecules[idx].pe = cro->objective_function(cro->molecules[idx].position);
        cro->buffer += rand_double(0.0, 1.0) * cro->molecules[idx].ke;
    }
    
    free(fitness);
    free(indices);
}

// Main optimization function
void CheRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    srand((unsigned int)time(NULL));
    CROOptimizer cro;
    cro.opt = opt;
    cro.population_size = opt->population_size;
    cro.objective_function = objective_function;
    
    initialize_molecules(&cro);
    
    for (int iter = 0; iter < cro.opt->max_iter; iter++) {
        evaluate_molecules(&cro);
        
        int i = 0;
        while (i < cro.population_size) {
            if (rand_double(0.0, 1.0) < CRO_MOLE_COLL && i < cro.population_size - 1) {
                int index1 = i;
                int index2 = i + 1;
                if (rand_double(0.0, 1.0) < 0.5) {
                    inter_molecular_collision(&cro, index1, index2);
                } else if (index1 < cro.population_size && index2 < cro.population_size) {
                    synthesis(&cro, index1, index2);
                }
                i += 2;
            } else {
                if (i < cro.population_size) {
                    if (rand_double(0.0, 1.0) < 0.5) {
                        on_wall_collision(&cro, i);
                    } else {
                        decomposition(&cro, i);
                    }
                }
                i++;
            }
        }
        
        elimination_phase_chero(&cro);
        printf("Iteration %d: Best Value = %f\n", iter + 1, cro.opt->best_solution.fitness);
    }
    
    // Cleanup
    for (int i = 0; i < cro.population_size; i++) {
        free(cro.molecules[i].position);
    }
    free(cro.molecules);
}
