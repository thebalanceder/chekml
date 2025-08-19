#include "LOA.h"
#include "generaloptimizer.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to generate a random double between min and max
double rand_double(double min, double max);

// Initialize Population
void initialize_population_loa(Optimizer *opt, LOAData *data) {
    int num_nomads = (int)(NOMAD_RATIO * opt->population_size);
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    data->genders = (unsigned char *)calloc(opt->population_size, sizeof(unsigned char));
    
    // Initialize indices and population
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
        opt->population[i].fitness = INFINITY;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
    
    // Shuffle indices
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = (int)(rand_double(0.0, 1.0) * (i + 1));
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Assign nomads and prides
    data->num_prides = (opt->population_size - num_nomads) / PRIDE_SIZE;
    data->prides = (int **)malloc(data->num_prides * sizeof(int *));
    data->pride_sizes = (int *)malloc(data->num_prides * sizeof(int));
    for (int i = 0; i < data->num_prides; i++) {
        data->pride_sizes[i] = PRIDE_SIZE;
        data->prides[i] = (int *)malloc(PRIDE_SIZE * sizeof(int));
        for (int j = 0; j < PRIDE_SIZE; j++) {
            data->prides[i][j] = indices[num_nomads + i * PRIDE_SIZE + j];
        }
    }
    
    data->nomads = (int *)malloc(num_nomads * sizeof(int));
    data->nomad_size = num_nomads;
    for (int i = 0; i < num_nomads; i++) {
        data->nomads[i] = indices[i];
    }

    // Assign genders
    for (int p = 0; p < data->num_prides; p++) {
        int num_females = (int)(FEMALE_RATIO * PRIDE_SIZE);
        for (int i = 0; i < num_females; i++) {
            int idx = data->prides[p][(int)(rand_double(0.0, 1.0) * PRIDE_SIZE)];
            data->genders[idx] = 1;
        }
    }
    int num_nomad_females = (int)((1.0 - FEMALE_RATIO) * num_nomads);
    for (int i = 0; i < num_nomad_females; i++) {
        int idx = data->nomads[(int)(rand_double(0.0, 1.0) * num_nomads)];
        data->genders[idx] = 1;
    }

    free(indices);
    enforce_bound_constraints(opt);
}

// Hunting Phase
void hunting_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function) {
    int *females = (int *)malloc(pride_size * sizeof(int));
    int num_females = 0;
    for (int i = 0; i < pride_size; i++) {
        if (data->genders[pride[i]]) {
            females[num_females++] = pride[i];
        }
    }
    if (num_females == 0) {
        free(females);
        return;
    }

    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    int *hunters = (int *)malloc(num_hunters * sizeof(int));
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(rand_double(0.0, 1.0) * num_females)];
    }

    double *prey = (double *)calloc(opt->dim, sizeof(double));
    for (int i = 0; i < num_hunters; i++) {
        for (int j = 0; j < opt->dim; j++) {
            prey[j] += opt->population[hunters[i]].position[j];
        }
    }
    for (int j = 0; j < opt->dim; j++) {
        prey[j] /= num_hunters;
    }

    for (int i = 0; i < num_hunters; i++) {
        int idx = hunters[i];
        double *new_pos = (double *)malloc(opt->dim * sizeof(double));
        double old_fitness = opt->population[idx].fitness;

        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[idx].position[j] < prey[j]) {
                new_pos[j] = opt->population[idx].position[j] + 
                             rand_double(0.0, 1.0) * (prey[j] - opt->population[idx].position[j]);
            } else {
                new_pos[j] = prey[j] + rand_double(0.0, 1.0) * (opt->population[idx].position[j] - prey[j]);
            }
        }

        double new_fitness = INFINITY;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] = new_pos[j];
        }
        enforce_bound_constraints(opt);
        new_fitness = objective_function(opt->population[idx].position);

        if (new_fitness < old_fitness) {
            double pi = old_fitness != 0 ? (old_fitness - new_fitness) / old_fitness : 1.0;
            for (int j = 0; j < opt->dim; j++) {
                prey[j] += rand_double(0.0, 1.0) * pi * (prey[j] - new_pos[j]);
            }
            opt->population[idx].fitness = new_fitness;
        }

        free(new_pos);
    }

    free(females);
    free(hunters);
    free(prey);
    enforce_bound_constraints(opt);
}

// Move to Safe Place Phase
void move_to_safe_place_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function) {
    int *females = (int *)malloc(pride_size * sizeof(int));
    int num_females = 0;
    for (int i = 0; i < pride_size; i++) {
        if (data->genders[pride[i]]) {
            females[num_females++] = pride[i];
        }
    }
    if (num_females == 0) {
        free(females);
        return;
    }

    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    int *hunters = (int *)malloc(num_hunters * sizeof(int));
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(rand_double(0.0, 1.0) * num_females)];
    }

    int *non_hunters = (int *)malloc(num_females * sizeof(int));
    int num_non_hunters = 0;
    for (int i = 0; i < num_females; i++) {
        int is_hunter = 0;
        for (int j = 0; j < num_hunters; j++) {
            if (females[i] == hunters[j]) {
                is_hunter = 1;
                break;
            }
        }
        if (!is_hunter) {
            non_hunters[num_non_hunters++] = females[i];
        }
    }

    for (int i = 0; i < num_non_hunters; i++) {
        int idx = non_hunters[i];
        int selected_idx = pride[(int)(rand_double(0.0, 1.0) * pride_size)];
        double *new_pos = (double *)malloc(opt->dim * sizeof(double));
        double d = 0.0;

        for (int j = 0; j < opt->dim; j++) {
            d += pow(opt->population[selected_idx].position[j] - opt->population[idx].position[j], 2);
        }
        d = sqrt(d);

        double *r1 = (double *)calloc(opt->dim, sizeof(double));
        double norm_r1 = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            r1[j] = opt->population[selected_idx].position[j] - opt->population[idx].position[j];
            norm_r1 += r1[j] * r1[j];
        }
        norm_r1 = norm_r1 > 0 ? sqrt(norm_r1) : 1e-10;
        for (int j = 0; j < opt->dim; j++) {
            r1[j] /= norm_r1;
        }

        double *r2 = (double *)malloc(opt->dim * sizeof(double));
        double norm_r2 = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            r2[j] = rand_double(-1.0, 1.0);
            norm_r2 += r2[j] * r2[j];
        }
        norm_r2 = norm_r2 > 0 ? sqrt(norm_r2) : 1e-10;
        for (int j = 0; j < opt->dim; j++) {
            r2[j] /= norm_r2;
        }

        double theta = (rand_double(0.0, 1.0) - 0.5) * M_PI;
        for (int j = 0; j < opt->dim; j++) {
            new_pos[j] = opt->population[idx].position[j] + 2.0 * d * rand_double(0.0, 1.0) * r1[j] +
                         (rand_double(-1.0, 1.0) * tan(theta) * d * r2[j]);
        }

        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] = new_pos[j];
        }
        enforce_bound_constraints(opt);
        double new_fitness = objective_function(opt->population[idx].position);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }

        free(r1);
        free(r2);
        free(new_pos);
    }

    free(females);
    free(hunters);
    free(non_hunters);
}

// Roaming Phase
void roaming_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function) {
    int *males = (int *)malloc(pride_size * sizeof(int));
    int num_males = 0;
    for (int i = 0; i < pride_size; i++) {
        if (!data->genders[pride[i]]) {
            males[num_males++] = pride[i];
        }
    }

    for (int i = 0; i < num_males; i++) {
        int idx = males[i];
        int num_visits = (int)(ROAMING_RATIO * pride_size);
        for (int v = 0; v < num_visits; v++) {
            int target_idx = pride[(int)(rand_double(0.0, 1.0) * pride_size)];
            double d = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                d += pow(opt->population[target_idx].position[j] - opt->population[idx].position[j], 2);
            }
            d = sqrt(d);

            double *direction = (double *)calloc(opt->dim, sizeof(double));
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                direction[j] = opt->population[target_idx].position[j] - opt->population[idx].position[j];
                norm += direction[j] * direction[j];
            }
            norm = norm > 0 ? sqrt(norm) : 1e-10;
            for (int j = 0; j < opt->dim; j++) {
                direction[j] /= norm;
            }

            double theta = (rand_double(0.0, 1.0) - 0.5) * M_PI / 3.0;
            double x = rand_double(0.0, 1.0) * 2.0 * d;

            double *new_pos = (double *)malloc(opt->dim * sizeof(double));
            for (int j = 0; j < opt->dim; j++) {
                new_pos[j] = opt->population[idx].position[j] + x * direction[j];
            }

            for (int j = 0; j < opt->dim; j++) {
                opt->population[idx].position[j] = new_pos[j];
            }
            enforce_bound_constraints(opt);
            double new_fitness = objective_function(opt->population[idx].position);
            if (new_fitness < opt->population[idx].fitness) {
                opt->population[idx].fitness = new_fitness;
            }

            free(direction);
            free(new_pos);
        }
    }

    free(males);
}

// Mating Phase
void loa_mating_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function) {
    int *females = (int *)malloc(pride_size * sizeof(int));
    int num_females = 0;
    for (int i = 0; i < pride_size; i++) {
        if (data->genders[pride[i]]) {
            females[num_females++] = pride[i];
        }
    }

    int num_mating = (int)(MATING_RATIO * num_females);
    int *mating_females = (int *)malloc(num_mating * sizeof(int));
    for (int i = 0; i < num_mating; i++) {
        mating_females[i] = females[(int)(rand_double(0.0, 1.0) * num_females)];
    }

    int *males = (int *)malloc(pride_size * sizeof(int));
    int num_males = 0;
    for (int i = 0; i < pride_size; i++) {
        if (!data->genders[pride[i]]) {
            males[num_males++] = pride[i];
        }
    }

    for (int i = 0; i < num_mating; i++) {
        if (num_males == 0) continue;
        int female_idx = mating_females[i];
        int male_idx = males[(int)(rand_double(0.0, 1.0) * num_males)];
        double beta = rand_double(0.4, 0.6);

        double *offspring = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) {
            offspring[j] = beta * opt->population[female_idx].position[j] + 
                           (1.0 - beta) * opt->population[male_idx].position[j];
            if (rand_double(0.0, 1.0) < MUTATION_PROB) {
                offspring[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }

        int worst_idx = 0;
        double worst_fitness = -INFINITY;
        for (int k = 0; k < opt->population_size; k++) {
            if (opt->population[k].fitness > worst_fitness) {
                worst_fitness = opt->population[k].fitness;
                worst_idx = k;
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            opt->population[worst_idx].position[j] = offspring[j];
        }
        enforce_bound_constraints(opt);
        opt->population[worst_idx].fitness = objective_function(opt->population[worst_idx].position);

        free(offspring);
    }

    free(females);
    free(mating_females);
    free(males);
}

// Nomad Movement Phase
void nomad_movement_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    double min_fitness = INFINITY, max_fitness = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        if (opt->population[i].fitness < min_fitness) min_fitness = opt->population[i].fitness;
        if (opt->population[i].fitness > max_fitness) max_fitness = opt->population[i].fitness;
    }

    for (int i = 0; i < data->nomad_size; i++) {
        int idx = data->nomads[i];
        double pr = (opt->population[idx].fitness - min_fitness) / (max_fitness - min_fitness + 1e-10);
        double *new_pos = (double *)malloc(opt->dim * sizeof(double));

        for (int j = 0; j < opt->dim; j++) {
            new_pos[j] = opt->population[idx].position[j];
            if (rand_double(0.0, 1.0) > pr) {
                new_pos[j] = opt->bounds[2 * j] + rand_double(0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }

        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] = new_pos[j];
        }
        enforce_bound_constraints(opt);
        double new_fitness = objective_function(opt->population[idx].position);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }

        free(new_pos);
    }
}

// Defense Phase
void defense_phase(Optimizer *opt, LOAData *data, int *pride, int pride_size, ObjectiveFunction objective_function) {
    int *males = (int *)malloc(pride_size * sizeof(int));
    int num_males = 0;
    for (int i = 0; i < pride_size; i++) {
        if (!data->genders[pride[i]]) {
            males[num_males++] = pride[i];
        }
    }
    if (num_males <= 1) {
        free(males);
        return;
    }

    int worst_male_idx = males[0];
    double worst_fitness = opt->population[males[0]].fitness;
    for (int i = 1; i < num_males; i++) {
        if (opt->population[males[i]].fitness > worst_fitness) {
            worst_fitness = opt->population[males[i]].fitness;
            worst_male_idx = males[i];
        }
    }

    for (int i = 0; i < pride_size; i++) {
        if (pride[i] == worst_male_idx) {
            for (int j = i; j < pride_size - 1; j++) {
                pride[j] = pride[j + 1];
            }
            pride_size--;
            break;
        }
    }

    int *new_nomads = (int *)malloc((data->nomad_size + 1) * sizeof(int));
    for (int i = 0; i < data->nomad_size; i++) {
        new_nomads[i] = data->nomads[i];
    }
    new_nomads[data->nomad_size] = worst_male_idx;
    data->nomad_size += 1;
    free(data->nomads);
    data->nomads = new_nomads;

    free(males);
}

// Immigration Phase
void immigration_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    for (int p = 0; p < data->num_prides; p++) {
        int *females = (int *)malloc(data->pride_sizes[p] * sizeof(int));
        int num_females = 0;
        for (int i = 0; i < data->pride_sizes[p]; i++) {
            if (data->genders[data->prides[p][i]]) {
                females[num_females++] = data->prides[p][i];
            }
        }

        int num_immigrants = (int)(IMMIGRATION_RATIO * num_females);
        for (int i = 0; i < num_immigrants; i++) {
            if (rand_double(0.0, 1.0) < 0.5) {
                int idx = females[(int)(rand_double(0.0, 1.0) * num_females)];
                for (int j = 0; j < data->pride_sizes[p]; j++) {
                    if (data->prides[p][j] == idx) {
                        for (int k = j; k < data->pride_sizes[p] - 1; k++) {
                            data->prides[p][k] = data->prides[p][k + 1];
                        }
                        data->pride_sizes[p]--;
                        break;
                    }
                }

                if (rand_double(0.0, 1.0) < 0.5 && data->num_prides > 1) {
                    int other_pride = (int)(rand_double(0.0, 1.0) * data->num_prides);
                    while (other_pride == p) {
                        other_pride = (int)(rand_double(0.0, 1.0) * data->num_prides);
                    }
                    int *new_pride = (int *)malloc((data->pride_sizes[other_pride] + 1) * sizeof(int));
                    for (int j = 0; j < data->pride_sizes[other_pride]; j++) {
                        new_pride[j] = data->prides[other_pride][j];
                    }
                    new_pride[data->pride_sizes[other_pride]] = idx;
                    data->pride_sizes[other_pride]++;
                    free(data->prides[other_pride]);
                    data->prides[other_pride] = new_pride;
                } else {
                    int *new_nomads = (int *)malloc((data->nomad_size + 1) * sizeof(int));
                    for (int j = 0; j < data->nomad_size; j++) {
                        new_nomads[j] = data->nomads[j];
                    }
                    new_nomads[data->nomad_size] = idx;
                    data->nomad_size++;
                    free(data->nomads);
                    data->nomads = new_nomads;
                }
            }
        }
        free(females);
    }

    int *nomad_females = (int *)malloc(data->nomad_size * sizeof(int));
    int num_nomad_females = 0;
    for (int i = 0; i < data->nomad_size; i++) {
        if (data->genders[data->nomads[i]]) {
            nomad_females[num_nomad_females++] = data->nomads[i];
        }
    }

    for (int i = 0; i < num_nomad_females; i++) {
        if (rand_double(0.0, 1.0) < 0.1 && data->num_prides > 0) {
            int idx = nomad_females[i];
            for (int j = 0; j < data->nomad_size; j++) {
                if (data->nomads[j] == idx) {
                    for (int k = j; k < data->nomad_size - 1; k++) {
                        data->nomads[k] = data->nomads[k + 1];
                    }
                    data->nomad_size--;
                    break;
                }
            }

            int random_pride = (int)(rand_double(0.0, 1.0) * data->num_prides);
            int *new_pride = (int *)malloc((data->pride_sizes[random_pride] + 1) * sizeof(int));
            for (int j = 0; j < data->pride_sizes[random_pride]; j++) {
                new_pride[j] = data->prides[random_pride][j];
            }
            new_pride[data->pride_sizes[random_pride]] = idx;
            data->pride_sizes[random_pride]++;
            free(data->prides[random_pride]);
            data->prides[random_pride] = new_pride;
        }
    }
    free(nomad_females);
}

// Population Control Phase
void population_control_phase(Optimizer *opt, LOAData *data) {
    for (int p = 0; p < data->num_prides; p++) {
        if (data->pride_sizes[p] > PRIDE_SIZE) {
            int *new_pride = (int *)malloc(PRIDE_SIZE * sizeof(int));
            for (int i = 0; i < PRIDE_SIZE; i++) {
                new_pride[i] = data->prides[p][i];
            }
            free(data->prides[p]);
            data->prides[p] = new_pride;
            data->pride_sizes[p] = PRIDE_SIZE;
        }
    }
}

// Main Optimization Function
void LOA_optimize(void *optimizer, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)optimizer;
    LOAData data = {0};

    srand(time(NULL));
    initialize_population_loa(opt, &data);

    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int p = 0; p < data.num_prides; p++) {
            hunting_phase(opt, &data, data.prides[p], data.pride_sizes[p], objective_function);
            move_to_safe_place_phase(opt, &data, data.prides[p], data.pride_sizes[p], objective_function);
            roaming_phase(opt, &data, data.prides[p], data.pride_sizes[p], objective_function);
            loa_mating_phase(opt, &data, data.prides[p], data.pride_sizes[p], objective_function);
            defense_phase(opt, &data, data.prides[p], data.pride_sizes[p], objective_function);
        }

        nomad_movement_phase(opt, &data, objective_function);
        immigration_phase(opt, &data, objective_function);
        population_control_phase(opt, &data);

        for (int i = 0; i < opt->population_size; i++) {
            double new_fitness = objective_function(opt->population[i].position);
            opt->population[i].fitness = new_fitness;
            if (new_fitness < opt->best_solution.fitness) {
                opt->best_solution.fitness = new_fitness;
                for (int j = 0; j < opt->dim; j++) {
                    opt->best_solution.position[j] = opt->population[i].position[j];
                }
            }
        }
        enforce_bound_constraints(opt);
    }

    // Cleanup
    for (int p = 0; p < data.num_prides; p++) {
        free(data.prides[p]);
    }
    free(data.prides);
    free(data.pride_sizes);
    free(data.nomads);
    free(data.genders);
}
