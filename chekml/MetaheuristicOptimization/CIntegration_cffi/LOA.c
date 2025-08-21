#include "LOA.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>

// Initialize Population
void loa_initialize_population(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    int num_nomads = (int)(NOMAD_RATIO * opt->population_size);
    data->num_prides = (opt->population_size - num_nomads) / PRIDE_SIZE;
    data->nomad_size = num_nomads;
    data->rng_state = (unsigned int)time(NULL);

    // Allocate buffers
    data->prides = (int *)malloc(data->num_prides * PRIDE_SIZE * sizeof(int));
    data->pride_sizes = (int *)malloc(data->num_prides * sizeof(int));
    data->nomads = (int *)malloc(num_nomads * sizeof(int));
    data->genders = (unsigned char *)calloc(opt->population_size, sizeof(unsigned char));
    data->temp_buffer = (double *)malloc(opt->dim * sizeof(double));
    data->females = (int *)malloc(PRIDE_SIZE * sizeof(int));
    data->hunters = (int *)malloc(PRIDE_SIZE * sizeof(int));
    data->non_hunters = (int *)malloc(PRIDE_SIZE * sizeof(int));
    data->males = (int *)malloc(PRIDE_SIZE * sizeof(int));
    data->mating_females = (int *)malloc(PRIDE_SIZE * sizeof(int));
    data->nomad_females = (int *)malloc(num_nomads * sizeof(int));

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = INFINITY;
        for (int j = 0; j < opt->dim; j++) {
            opt->population[i].position[j] = opt->bounds[2 * j] + 
                                            loa_rand_double(data, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
        // Compute initial fitness
        enforce_bound_constraints(opt);
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Assign prides and nomads
    int *indices = (int *)malloc(opt->population_size * sizeof(int));
    for (int i = 0; i < opt->population_size; i++) {
        indices[i] = i;
    }
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = (int)(loa_rand_double(data, 0.0, 1.0) * (i + 1));
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = 0; i < data->num_prides; i++) {
        data->pride_sizes[i] = PRIDE_SIZE;
        for (int j = 0; j < PRIDE_SIZE; j++) {
            data->prides[i * PRIDE_SIZE + j] = indices[num_nomads + i * PRIDE_SIZE + j];
        }
    }
    for (int i = 0; i < num_nomads; i++) {
        data->nomads[i] = indices[i];
    }
    free(indices);

    // Assign genders
    for (int p = 0; p < data->num_prides; p++) {
        int num_females = (int)(FEMALE_RATIO * PRIDE_SIZE);
        for (int i = 0; i < num_females; i++) {
            int idx = data->prides[p * PRIDE_SIZE + (int)(loa_rand_double(data, 0.0, 1.0) * PRIDE_SIZE)];
            data->genders[idx] = 1;
        }
    }
    int num_nomad_females = (int)((1.0 - FEMALE_RATIO) * num_nomads);
    for (int i = 0; i < num_nomad_females; i++) {
        int idx = data->nomads[(int)(loa_rand_double(data, 0.0, 1.0) * num_nomads)];
        data->genders[idx] = 1;
    }
}

// Hunting Phase
void hunting_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *females = data->females;
    int num_females = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (data->genders[idx]) {
            females[num_females++] = idx;
        }
    }
    if (num_females == 0) return;

    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    int *hunters = data->hunters;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(loa_rand_double(data, 0.0, 1.0) * num_females)];
    }

    double *prey = data->temp_buffer;
    for (int j = 0; j < opt->dim; j++) prey[j] = 0.0;
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
        double old_fitness = opt->population[idx].fitness;

        for (int j = 0; j < opt->dim; j++) {
            double pos = opt->population[idx].position[j];
            double p = prey[j];
            opt->population[idx].position[j] = pos + loa_rand_double(data, 0.0, 1.0) * (p - pos) * (pos < p ? 1.0 : -1.0);
        }
        enforce_bound_constraints(opt);
        double new_fitness = objective_function(opt->population[idx].position);

        if (new_fitness < old_fitness) {
            double pi = old_fitness != 0 ? (old_fitness - new_fitness) / old_fitness : 1.0;
            for (int j = 0; j < opt->dim; j++) {
                prey[j] += loa_rand_double(data, 0.0, 1.0) * pi * (prey[j] - opt->population[idx].position[j]);
            }
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Move to Safe Place Phase
void move_to_safe_place_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *females = data->females;
    int num_females = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (data->genders[idx]) {
            females[num_females++] = idx;
        }
    }
    if (num_females == 0) return;

    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    int *hunters = data->hunters;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(loa_rand_double(data, 0.0, 1.0) * num_females)];
    }

    int *non_hunters = data->non_hunters;
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
        int selected_idx = data->prides[pride_idx * PRIDE_SIZE + (int)(loa_rand_double(data, 0.0, 1.0) * data->pride_sizes[pride_idx])];
        double d = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = opt->population[selected_idx].position[j] - opt->population[idx].position[j];
            d += diff * diff;
        }
        d = sqrt(d);

        double *r1 = data->temp_buffer;
        double norm_r1 = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            r1[j] = opt->population[selected_idx].position[j] - opt->population[idx].position[j];
            norm_r1 += r1[j] * r1[j];
        }
        norm_r1 = norm_r1 > 0 ? sqrt(norm_r1) : 1e-10;
        for (int j = 0; j < opt->dim; j++) {
            r1[j] /= norm_r1;
        }

        double theta = (loa_rand_double(data, 0.0, 1.0) - 0.5) * M_PI;
        double tan_theta = tan(theta);
        for (int j = 0; j < opt->dim; j++) {
            opt->population[idx].position[j] += 2.0 * d * loa_rand_double(data, 0.0, 1.0) * r1[j] +
                                               loa_rand_double(data, -1.0, 1.0) * tan_theta * d * r1[j];
        }
        enforce_bound_constraints(opt);
        double new_fitness = objective_function(opt->population[idx].position);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Roaming Phase
void roaming_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *males = data->males;
    int num_males = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (!data->genders[idx]) {
            males[num_males++] = idx;
        }
    }

    int num_visits = (int)(ROAMING_RATIO * data->pride_sizes[pride_idx]);
    for (int i = 0; i < num_males; i++) {
        int idx = males[i];
        for (int v = 0; v < num_visits; v++) {
            int target_idx = data->prides[pride_idx * PRIDE_SIZE + (int)(loa_rand_double(data, 0.0, 1.0) * data->pride_sizes[pride_idx])];
            double d = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                double diff = opt->population[target_idx].position[j] - opt->population[idx].position[j];
                d += diff * diff;
            }
            d = sqrt(d);

            double *direction = data->temp_buffer;
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                direction[j] = opt->population[target_idx].position[j] - opt->population[idx].position[j];
                norm += direction[j] * direction[j];
            }
            norm = norm > 0 ? sqrt(norm) : 1e-10;
            for (int j = 0; j < opt->dim; j++) {
                direction[j] /= norm;
            }

            double theta = loa_rand_double(data, -0.5, 0.5) * M_PI / 3.0;
            double x = loa_rand_double(data, 0.0, 1.0) * 2.0 * d;
            for (int j = 0; j < opt->dim; j++) {
                opt->population[idx].position[j] += x * direction[j];
            }
            enforce_bound_constraints(opt);
            double new_fitness = objective_function(opt->population[idx].position);
            if (new_fitness < opt->population[idx].fitness) {
                opt->population[idx].fitness = new_fitness;
            }
        }
    }
}

// Mating Phase
void loa_mating_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *females = data->females;
    int num_females = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (data->genders[idx]) {
            females[num_females++] = idx;
        }
    }

    int num_mating = (int)(MATING_RATIO * num_females);
    int *mating_females = data->mating_females;
    for (int i = 0; i < num_mating; i++) {
        mating_females[i] = females[(int)(loa_rand_double(data, 0.0, 1.0) * num_females)];
    }

    int *males = data->males;
    int num_males = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (!data->genders[idx]) {
            males[num_males++] = idx;
        }
    }

    for (int i = 0; i < num_mating; i++) {
        if (num_males == 0) continue;
        int female_idx = mating_females[i];
        int male_idx = males[(int)(loa_rand_double(data, 0.0, 1.0) * num_males)];
        double beta = loa_rand_double(data, 0.4, 0.6);

        double *offspring = data->temp_buffer;
        for (int j = 0; j < opt->dim; j++) {
            offspring[j] = beta * opt->population[female_idx].position[j] + 
                           (1.0 - beta) * opt->population[male_idx].position[j];
            if (loa_rand_double(data, 0.0, 1.0) < MUTATION_PROB) {
                offspring[j] = opt->bounds[2 * j] + loa_rand_double(data, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
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
    }
}

// Nomad Movement Phase
void nomad_movement_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    double min_fitness = INFINITY, max_fitness = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double f = opt->population[i].fitness;
        if (f < min_fitness) min_fitness = f;
        if (f > max_fitness) max_fitness = f;
    }

    for (int i = 0; i < data->nomad_size; i++) {
        int idx = data->nomads[i];
        double pr = (opt->population[idx].fitness - min_fitness) / (max_fitness - min_fitness + 1e-10);

        for (int j = 0; j < opt->dim; j++) {
            if (loa_rand_double(data, 0.0, 1.0) > pr) {
                opt->population[idx].position[j] = opt->bounds[2 * j] + 
                                                 loa_rand_double(data, 0.0, 1.0) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
        }
        enforce_bound_constraints(opt);
        double new_fitness = objective_function(opt->population[idx].position);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Defense Phase
void defense_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *males = data->males;
    int num_males = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        int idx = data->prides[pride_idx * PRIDE_SIZE + i];
        if (!data->genders[idx]) {
            males[num_males++] = idx;
        }
    }
    if (num_males <= 1) return;

    int worst_male_idx = males[0];
    double worst_fitness = opt->population[males[0]].fitness;
    for (int i = 1; i < num_males; i++) {
        double f = opt->population[males[i]].fitness;
        if (f > worst_fitness) {
            worst_fitness = f;
            worst_male_idx = males[i];
        }
    }

    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (data->prides[pride_idx * PRIDE_SIZE + i] == worst_male_idx) {
            for (int j = i; j < data->pride_sizes[pride_idx] - 1; j++) {
                data->prides[pride_idx * PRIDE_SIZE + j] = data->prides[pride_idx * PRIDE_SIZE + j + 1];
            }
            data->pride_sizes[pride_idx]--;
            break;
        }
    }

    int *new_nomads = (int *)malloc((data->nomad_size + 1) * sizeof(int));
    for (int i = 0; i < data->nomad_size; i++) {
        new_nomads[i] = data->nomads[i];
    }
    new_nomads[data->nomad_size] = worst_male_idx;
    data->nomad_size++;
    free(data->nomads);
    data->nomads = new_nomads;
}

// Immigration Phase
void immigration_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    for (int p = 0; p < data->num_prides; p++) {
        int *females = data->females;
        int num_females = 0;
        for (int i = 0; i < data->pride_sizes[p]; i++) {
            int idx = data->prides[p * PRIDE_SIZE + i];
            if (data->genders[idx]) {
                females[num_females++] = idx;
            }
        }

        int num_immigrants = (int)(IMMIGRATION_RATIO * num_females);
        for (int i = 0; i < num_immigrants; i++) {
            if (loa_rand_double(data, 0.0, 1.0) < 0.5) {
                int idx = females[(int)(loa_rand_double(data, 0.0, 1.0) * num_females)];
                for (int j = 0; j < data->pride_sizes[p]; j++) {
                    if (data->prides[p * PRIDE_SIZE + j] == idx) {
                        for (int k = j; k < data->pride_sizes[p] - 1; k++) {
                            data->prides[p * PRIDE_SIZE + k] = data->prides[p * PRIDE_SIZE + k + 1];
                        }
                        data->pride_sizes[p]--;
                        break;
                    }
                }

                if (loa_rand_double(data, 0.0, 1.0) < 0.5 && data->num_prides > 1) {
                    int other_pride = (int)(loa_rand_double(data, 0.0, 1.0) * data->num_prides);
                    while (other_pride == p) {
                        other_pride = (int)(loa_rand_double(data, 0.0, 1.0) * data->num_prides);
                    }
                    if (data->pride_sizes[other_pride] < PRIDE_SIZE) {
                        data->prides[other_pride * PRIDE_SIZE + data->pride_sizes[other_pride]] = idx;
                        data->pride_sizes[other_pride]++;
                    }
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
    }

    int *nomad_females = data->nomad_females;
    int num_nomad_females = 0;
    for (int i = 0; i < data->nomad_size; i++) {
        if (data->genders[data->nomads[i]]) {
            nomad_females[num_nomad_females++] = data->nomads[i];
        }
    }

    for (int i = 0; i < num_nomad_females; i++) {
        if (loa_rand_double(data, 0.0, 1.0) < 0.1 && data->num_prides > 0) {
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

            int random_pride = (int)(loa_rand_double(data, 0.0, 1.0) * data->num_prides);
            if (data->pride_sizes[random_pride] < PRIDE_SIZE) {
                data->prides[random_pride * PRIDE_SIZE + data->pride_sizes[random_pride]] = idx;
                data->pride_sizes[random_pride]++;
            }
        }
    }
}

// Population Control Phase
void population_control_phase(Optimizer *opt, LOAData *data) {
    for (int p = 0; p < data->num_prides; p++) {
        if (data->pride_sizes[p] > PRIDE_SIZE) {
            data->pride_sizes[p] = PRIDE_SIZE;
        }
    }
}

// Main Optimization Function
void LOA_optimize(void *optimizer, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)optimizer;
    LOAData data = {0};

    loa_initialize_population(opt, &data, objective_function);

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = opt->population[i].position[j];
            }
        }
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        for (int p = 0; p < data.num_prides; p++) {
            hunting_phase(opt, &data, p, objective_function);
            move_to_safe_place_phase(opt, &data, p, objective_function);
            roaming_phase(opt, &data, p, objective_function);
            loa_mating_phase(opt, &data, p, objective_function);
            defense_phase(opt, &data, p, objective_function);
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

        // Debug logging
        if (iter % 10 == 0) {
            printf("Iteration %d: Best fitness = %f, Best solution = [", iter, opt->best_solution.fitness);
            for (int j = 0; j < opt->dim; j++) {
                printf("%f", opt->best_solution.position[j]);
                if (j < opt->dim - 1) printf(", ");
            }
            printf("]\n");
        }
    }

    // Cleanup
    free(data.prides);
    free(data.pride_sizes);
    free(data.nomads);
    free(data.genders);
    free(data.temp_buffer);
    free(data.females);
    free(data.hunters);
    free(data.non_hunters);
    free(data.males);
    free(data.mating_females);
    free(data.nomad_females);
}
