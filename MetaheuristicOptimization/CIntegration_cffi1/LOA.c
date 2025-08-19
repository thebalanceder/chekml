#include "LOA.h"
#include "generaloptimizer.h"
#include <string.h>
#include <time.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Initialize Population
void loa_initialize_population(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    if (!opt || opt->population_size <= 0 || opt->dim <= 0) {
        fprintf(stderr, "Invalid optimizer parameters\n");
        exit(1);
    }

    int num_nomads = (int)(NOMAD_RATIO * opt->population_size);
    data->num_prides = (opt->population_size - num_nomads) / PRIDE_SIZE;
    data->nomad_size = num_nomads;
    data->nomad_capacity = num_nomads + opt->population_size / 2;
    data->rng_state = (unsigned int)time(NULL) ^ ((unsigned int)time(NULL) >> 3);

    // Allocate contiguous memory
    size_t total_size = (data->num_prides * PRIDE_SIZE + data->nomad_capacity + opt->population_size +
                         4 * PRIDE_SIZE + num_nomads) * sizeof(int) +
                        opt->population_size * sizeof(unsigned char) +
                        opt->dim * sizeof(double) +
                        opt->population_size * sizeof(int);
    void *buffer = malloc(total_size);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    char *ptr = (char *)buffer;
    data->prides = (int *)(ptr); ptr += data->num_prides * PRIDE_SIZE * sizeof(int);
    data->pride_sizes = (int *)(ptr); ptr += data->num_prides * sizeof(int);
    data->nomads = (int *)(ptr); ptr += data->nomad_capacity * sizeof(int);
    data->genders = (unsigned char *)(ptr); ptr += opt->population_size * sizeof(unsigned char);
    data->temp_buffer = (double *)(ptr); ptr += opt->dim * sizeof(double);
    data->females = (int *)(ptr); ptr += PRIDE_SIZE * sizeof(int);
    data->hunters = (int *)(ptr); ptr += PRIDE_SIZE * sizeof(int);
    data->non_hunters = (int *)(ptr); ptr += PRIDE_SIZE * sizeof(int);
    data->males = (int *)(ptr); ptr += PRIDE_SIZE * sizeof(int);
    data->mating_females = (int *)(ptr); ptr += PRIDE_SIZE * sizeof(int);
    data->nomad_females = (int *)(ptr); ptr += num_nomads * sizeof(int);
    data->index_buffer = (int *)(ptr);

    memset(data->genders, 0, opt->population_size * sizeof(unsigned char));
    memset(data->pride_sizes, 0, data->num_prides * sizeof(int));

    // Initialize population
    for (int i = 0; i < opt->population_size; i++) {
        opt->population[i].fitness = INFINITY;
        for (int j = 0; j < opt->dim; j++) {
            double min = opt->bounds[2 * j], range = opt->bounds[2 * j + 1] - min;
            opt->population[i].position[j] = min + range * loa_fast_rand(data);
            if (opt->population[i].position[j] < min || opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                fprintf(stderr, "Bound violation at init: pos[%d][%d] = %f\n", i, j, opt->population[i].position[j]);
                exit(1);
            }
        }
        opt->population[i].fitness = objective_function(opt->population[i].position);
    }

    // Assign prides and nomads
    for (int i = 0; i < opt->population_size; i++) data->index_buffer[i] = i;
    for (int i = opt->population_size - 1; i > 0; i--) {
        int j = (int)(loa_fast_rand(data) * (i + 1));
        int temp = data->index_buffer[i];
        data->index_buffer[i] = data->index_buffer[j];
        data->index_buffer[j] = temp;
    }

    for (int i = 0; i < data->num_prides; i++) {
        data->pride_sizes[i] = PRIDE_SIZE;
        memcpy(&data->prides[i * PRIDE_SIZE], &data->index_buffer[num_nomads + i * PRIDE_SIZE], PRIDE_SIZE * sizeof(int));
    }
    memcpy(data->nomads, data->index_buffer, num_nomads * sizeof(int));

    // Assign genders
    for (int p = 0; p < data->num_prides; p++) {
        int num_females = (int)(FEMALE_RATIO * PRIDE_SIZE);
        for (int i = 0; i < num_females; i++) {
            int idx = data->prides[p * PRIDE_SIZE + (int)(loa_fast_rand(data) * PRIDE_SIZE)];
            if (idx >= opt->population_size) {
                fprintf(stderr, "Invalid pride index: %d\n", idx);
                exit(1);
            }
            data->genders[idx] = 1;
        }
    }
    int num_nomad_females = (int)((1.0f - FEMALE_RATIO) * num_nomads);
    for (int i = 0; i < num_nomad_females; i++) {
        int idx = data->nomads[(int)(loa_fast_rand(data) * num_nomads)];
        if (idx >= opt->population_size) {
            fprintf(stderr, "Invalid nomad index: %d\n", idx);
            exit(1);
        }
        data->genders[idx] = 1;
    }
}

// Hunting Phase
void hunting_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    if (pride_idx >= data->num_prides) {
        fprintf(stderr, "Invalid pride index: %d\n", pride_idx);
        exit(1);
    }
    int *females = data->females;
    int num_females = 0;
    int *pride = &data->prides[pride_idx * PRIDE_SIZE];
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (pride[i] >= opt->population_size) {
            fprintf(stderr, "Invalid pride member index: %d\n", pride[i]);
            exit(1);
        }
        if (data->genders[pride[i]]) females[num_females++] = pride[i];
    }
    if (num_females == 0) return;

    int num_hunters = num_females > 1 ? num_females / 2 : 1;
    int *hunters = data->hunters;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(loa_fast_rand(data) * num_females)];
        if (hunters[i] >= opt->population_size) {
            fprintf(stderr, "Invalid hunter index: %d\n", hunters[i]);
            exit(1);
        }
    }

    double *prey = data->temp_buffer;
    memset(prey, 0, opt->dim * sizeof(double));
    for (int i = 0; i < num_hunters; i++) {
        double *pos = opt->population[hunters[i]].position;
        for (int j = 0; j < opt->dim; j++) prey[j] += pos[j];
    }
    for (int j = 0; j < opt->dim; j++) prey[j] /= num_hunters;

    for (int i = 0; i < num_hunters; i++) {
        int idx = hunters[i];
        double old_fitness = opt->population[idx].fitness;
        double *pos = opt->population[idx].position;

        for (int j = 0; j < opt->dim; j++) {
            double p = prey[j], x = pos[j];
            pos[j] = x + loa_fast_rand(data) * (p - x) * (x < p ? 1.0 : -1.0);
            pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], pos[j]));
        }
        double new_fitness = objective_function(pos);

        if (new_fitness < old_fitness) {
            double pi = old_fitness != 0 ? (old_fitness - new_fitness) / old_fitness : 1.0;
            for (int j = 0; j < opt->dim; j++) {
                prey[j] += loa_fast_rand(data) * pi * (prey[j] - pos[j]);
            }
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Move to Safe Place Phase
void move_to_safe_place_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *females = data->females;
    int num_females = 0;
    int *pride = &data->prides[pride_idx * PRIDE_SIZE];
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (data->genders[pride[i]]) females[num_females++] = pride[i];
    }
    if (num_females == 0) return;

    int num_hunters = num_females > 1 ? num_females / 2 : 1;
    int *hunters = data->hunters;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(loa_fast_rand(data) * num_females)];
    }

    int *non_hunters = data->non_hunters;
    int num_non_hunters = 0;
    for (int i = 0; i < num_females; i++) {
        int is_hunter = 0;
        for (int j = 0; j < num_hunters; j++) {
            if (females[i] == hunters[j]) { is_hunter = 1; break; }
        }
        if (!is_hunter) non_hunters[num_non_hunters++] = females[i];
    }

    for (int i = 0; i < num_non_hunters; i++) {
        int idx = non_hunters[i];
        int sel_idx = pride[(int)(loa_fast_rand(data) * data->pride_sizes[pride_idx])];
        double *pos = opt->population[idx].position;
        double *sel_pos = opt->population[sel_idx].position;

        double d = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            double diff = sel_pos[j] - pos[j];
            d += diff * diff;
        }
        d = sqrt(d);

        double *r1 = data->temp_buffer;
        double norm_r1 = 0.0;
        for (int j = 0; j < opt->dim; j++) {
            r1[j] = sel_pos[j] - pos[j];
            norm_r1 += r1[j] * r1[j];
        }
        norm_r1 = norm_r1 > 0 ? sqrt(norm_r1) : 1e-10;
        double inv_norm = 1.0 / norm_r1;
        for (int j = 0; j < opt->dim; j++) r1[j] *= inv_norm;

        double theta = (loa_fast_rand(data) - 0.5) * M_PI;
        double tan_theta = tan(theta);
        for (int j = 0; j < opt->dim; j++) {
            pos[j] += (2.0 * d * loa_fast_rand(data) + loa_fast_rand(data) * tan_theta * d) * r1[j];
            pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], pos[j]));
        }
        double new_fitness = objective_function(pos);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Roaming Phase
void roaming_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *males = data->males;
    int num_males = 0;
    int *pride = &data->prides[pride_idx * PRIDE_SIZE];
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (!data->genders[pride[i]]) males[num_males++] = pride[i];
    }

    int num_visits = (int)(ROAMING_RATIO * data->pride_sizes[pride_idx]);
    for (int i = 0; i < num_males; i++) {
        int idx = males[i];
        double *pos = opt->population[idx].position;
        for (int v = 0; v < num_visits; v++) {
            int target_idx = pride[(int)(loa_fast_rand(data) * data->pride_sizes[pride_idx])];
            double *target_pos = opt->population[target_idx].position;

            double d = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                double diff = target_pos[j] - pos[j];
                d += diff * diff;
            }
            d = sqrt(d);

            double *direction = data->temp_buffer;
            double norm = 0.0;
            for (int j = 0; j < opt->dim; j++) {
                direction[j] = target_pos[j] - pos[j];
                norm += direction[j] * direction[j];
            }
            norm = norm > 0 ? sqrt(norm) : 1e-10;
            double inv_norm = 1.0 / norm;
            for (int j = 0; j < opt->dim; j++) direction[j] *= inv_norm;

            double x = loa_fast_rand(data) * 2.0 * d;
            for (int j = 0; j < opt->dim; j++) {
                pos[j] += x * direction[j];
                pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], pos[j]));
            }
            double new_fitness = objective_function(pos);
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
    int *pride = &data->prides[pride_idx * PRIDE_SIZE];
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (data->genders[pride[i]]) females[num_females++] = pride[i];
    }

    int num_mating = (int)(MATING_RATIO * num_females);
    int *mating_females = data->mating_females;
    for (int i = 0; i < num_mating; i++) {
        mating_females[i] = females[(int)(loa_fast_rand(data) * num_females)];
    }

    int *males = data->males;
    int num_males = 0;
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (!data->genders[pride[i]]) males[num_males++] = pride[i];
    }

    int worst_idx = 0;
    double worst_fitness = -INFINITY;
    for (int k = 0; k < opt->population_size; k++) {
        if (opt->population[k].fitness > worst_fitness) {
            worst_fitness = opt->population[k].fitness;
            worst_idx = k;
        }
    }

    for (int i = 0; i < num_mating; i++) {
        if (num_males == 0) continue;
        int female_idx = mating_females[i];
        int male_idx = males[(int)(loa_fast_rand(data) * num_males)];
        double beta = 0.5;

        double *offspring = data->temp_buffer;
        double *f_pos = opt->population[female_idx].position;
        double *m_pos = opt->population[male_idx].position;
        for (int j = 0; j < opt->dim; j++) {
            offspring[j] = beta * f_pos[j] + (1.0 - beta) * m_pos[j];
            if (loa_fast_rand(data) < MUTATION_PROB) {
                double min = opt->bounds[2 * j], range = opt->bounds[2 * j + 1] - min;
                offspring[j] = min + range * loa_fast_rand(data);
            }
        }

        double *w_pos = opt->population[worst_idx].position;
        for (int j = 0; j < opt->dim; j++) {
            w_pos[j] = offspring[j];
            w_pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], w_pos[j]));
        }
        opt->population[worst_idx].fitness = objective_function(w_pos);
    }
}

// Nomad Movement Phase
void nomad_movement_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    double min_fitness = INFINITY, max_fitness = -INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double f = opt->population[i].fitness;
        min_fitness = fmin(min_fitness, f);
        max_fitness = fmax(max_fitness, f);
    }
    double fitness_range = max_fitness - min_fitness + 1e-10;

    for (int i = 0; i < data->nomad_size; i++) {
        int idx = data->nomads[i];
        if (idx >= opt->population_size) {
            fprintf(stderr, "Invalid nomad index: %d\n", idx);
            exit(1);
        }
        double pr = (opt->population[idx].fitness - min_fitness) / fitness_range;
        double *pos = opt->population[idx].position;

        for (int j = 0; j < opt->dim; j++) {
            if (loa_fast_rand(data) > pr) {
                double min = opt->bounds[2 * j], range = opt->bounds[2 * j + 1] - min;
                pos[j] = min + range * loa_fast_rand(data);
            }
            pos[j] = fmax(opt->bounds[2 * j], fmin(opt->bounds[2 * j + 1], pos[j]));
        }
        double new_fitness = objective_function(pos);
        if (new_fitness < opt->population[idx].fitness) {
            opt->population[idx].fitness = new_fitness;
        }
    }
}

// Defense Phase
void defense_phase(Optimizer *opt, LOAData *data, int pride_idx, ObjectiveFunction objective_function) {
    int *males = data->males;
    int num_males = 0;
    int *pride = &data->prides[pride_idx * PRIDE_SIZE];
    for (int i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (!data->genders[pride[i]]) males[num_males++] = pride[i];
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

    int i;
    for (i = 0; i < data->pride_sizes[pride_idx]; i++) {
        if (pride[i] == worst_male_idx) break;
    }
    for (; i < data->pride_sizes[pride_idx] - 1; i++) {
        pride[i] = pride[i + 1];
    }
    data->pride_sizes[pride_idx]--;

    if (data->nomad_size >= data->nomad_capacity) {
        data->nomad_capacity += opt->population_size / 4;
        int *new_nomads = realloc(data->nomads, data->nomad_capacity * sizeof(int));
        if (!new_nomads) {
            fprintf(stderr, "Nomad realloc failed\n");
            exit(1);
        }
        data->nomads = new_nomads;
    }
    data->nomads[data->nomad_size++] = worst_male_idx;
}

// Immigration Phase
void immigration_phase(Optimizer *opt, LOAData *data, ObjectiveFunction objective_function) {
    for (int p = 0; p < data->num_prides; p++) {
        int *females = data->females;
        int num_females = 0;
        int *pride = &data->prides[p * PRIDE_SIZE];
        for (int i = 0; i < data->pride_sizes[p]; i++) {
            if (data->genders[pride[i]]) females[num_females++] = pride[i];
        }

        int num_immigrants = (int)(IMMIGRATION_RATIO * num_females);
        for (int i = 0; i < num_immigrants; i++) {
            if (loa_fast_rand(data) < 0.5) {
                int idx = females[(int)(loa_fast_rand(data) * num_females)];
                int j;
                for (j = 0; j < data->pride_sizes[p]; j++) {
                    if (pride[j] == idx) break;
                }
                for (; j < data->pride_sizes[p] - 1; j++) {
                    pride[j] = pride[j + 1];
                }
                data->pride_sizes[p]--;

                if (loa_fast_rand(data) < 0.5 && data->num_prides > 1) {
                    int other_pride = (int)(loa_fast_rand(data) * data->num_prides);
                    while (other_pride == p) other_pride = (int)(loa_fast_rand(data) * data->num_prides);
                    if (data->pride_sizes[other_pride] < PRIDE_SIZE) {
                        data->prides[other_pride * PRIDE_SIZE + data->pride_sizes[other_pride]++] = idx;
                    }
                } else {
                    if (data->nomad_size >= data->nomad_capacity) {
                        data->nomad_capacity += opt->population_size / 4;
                        int *new_nomads = realloc(data->nomads, data->nomad_capacity * sizeof(int));
                        if (!new_nomads) {
                            fprintf(stderr, "Nomad realloc failed\n");
                            exit(1);
                        }
                        data->nomads = new_nomads;
                    }
                    data->nomads[data->nomad_size++] = idx;
                }
            }
        }
    }

    int *nomad_females = data->nomad_females;
    int num_nomad_females = 0;
    for (int i = 0; i < data->nomad_size; i++) {
        if (data->genders[data->nomads[i]]) nomad_females[num_nomad_females++] = data->nomads[i];
    }

    for (int i = 0; i < num_nomad_females; i++) {
        if (loa_fast_rand(data) < 0.1 && data->num_prides > 0) {
            int idx = nomad_females[i];
            int j;
            for (j = 0; j < data->nomad_size; j++) {
                if (data->nomads[j] == idx) break;
            }
            for (; j < data->nomad_size - 1; j++) {
                data->nomads[j] = data->nomads[j + 1];
            }
            data->nomad_size--;

            int random_pride = (int)(loa_fast_rand(data) * data->num_prides);
            if (data->pride_sizes[random_pride] < PRIDE_SIZE) {
                data->prides[random_pride * PRIDE_SIZE + data->pride_sizes[random_pride]++] = idx;
            }
        }
    }
}

// Population Control Phase
void population_control_phase(Optimizer *opt, LOAData *data) {
    for (int p = 0; p < data->num_prides; p++) {
        if (data->pride_sizes[p] > PRIDE_SIZE) data->pride_sizes[p] = PRIDE_SIZE;
    }
}

// Main Optimization Function
void LOA_optimize(void *optimizer, ObjectiveFunction objective_function) {
    Optimizer *opt = (Optimizer *)optimizer;
    if (!opt || !objective_function) {
        fprintf(stderr, "Invalid optimizer or objective function\n");
        exit(1);
    }

    LOAData data = {0};
    loa_initialize_population(opt, &data, objective_function);

    opt->best_solution.fitness = INFINITY;
    for (int i = 0; i < opt->population_size; i++) {
        double fitness = opt->population[i].fitness;
        if (fitness < opt->best_solution.fitness) {
            opt->best_solution.fitness = fitness;
            memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
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
                memcpy(opt->best_solution.position, opt->population[i].position, opt->dim * sizeof(double));
            }
        }

        if (iter % 10 == 0) {
            printf("Iteration %d: Best fitness = %f\n", iter, opt->best_solution.fitness);
        }
    }

    free(data.prides);
}
