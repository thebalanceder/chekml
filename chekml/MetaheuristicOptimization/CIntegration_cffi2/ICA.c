#include "ICA.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Inline random double generator
static inline double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Static costs pointer for qsort comparison
static double *sort_costs;

// Comparison function for qsort
static int compare_indices(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return sort_costs[ia] > sort_costs[ib] ? 1 : (sort_costs[ia] < sort_costs[ib] ? -1 : 0);
}

// Initialize empires
void create_initial_empires(Optimizer *opt, ICAParams *params, Empire **empires, int *num_empires, ICAOptimizerData *data) {
    if (!opt || !params || !empires || !num_empires || !opt->population || !opt->bounds) {
        printf("Invalid arguments in create_initial_empires\n");
        exit(1);
    }

    // Adjust number of imperialists
    int effective_imperialists = NUM_IMPERIALISTS;
    int num_colonies_total = opt->population_size - effective_imperialists;
    if (num_colonies_total < effective_imperialists) {
        effective_imperialists = opt->population_size / 2;
        if (effective_imperialists < 1) effective_imperialists = 1;
        num_colonies_total = opt->population_size - effective_imperialists;
    }
    *num_empires = effective_imperialists;

    *empires = (Empire *)calloc(*num_empires, sizeof(Empire));
    if (!*empires) {
        printf("Memory allocation failed for empires\n");
        exit(1);
    }

    // Pre-allocate temporary arrays on stack
    double costs[NUM_COUNTRIES];
    int indices[NUM_COUNTRIES];
    if (opt->population_size > NUM_COUNTRIES) {
        printf("Population size %d exceeds max %d\n", opt->population_size, NUM_COUNTRIES);
        free(*empires);
        exit(1);
    }

    // Evaluate population costs
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            printf("Invalid population position at index %d\n", i);
            free(*empires);
            exit(1);
        }
        costs[i] = params->objective_function(opt->population[i].position);
        indices[i] = i;
        data->fes++;
    }

    // Sort by cost
    sort_costs = costs;
    qsort(indices, opt->population_size, sizeof(int), compare_indices);
    sort_costs = NULL;

    // Compute normalized power
    double max_cost = costs[indices[0]];
    for (int i = 1; i < effective_imperialists; i++) {
        if (costs[indices[i]] > max_cost) max_cost = costs[indices[i]];
    }

    double power_sum = 0.0;
    double power[NUM_IMPERIALISTS];
    for (int i = 0; i < effective_imperialists; i++) {
        power[i] = max_cost - costs[indices[i]];
        power_sum += power[i];
    }
    if (power_sum == 0.0) {
        printf("Power sum is zero\n");
        free(*empires);
        exit(1);
    }

    // Assign colonies
    int num_colonies[NUM_IMPERIALISTS];
    int remaining = num_colonies_total;
    for (int i = 0; i < effective_imperialists - 1; i++) {
        num_colonies[i] = (int)(power[i] / power_sum * num_colonies_total);
        if (num_colonies[i] > remaining) num_colonies[i] = remaining;
        if (num_colonies[i] < 0) num_colonies[i] = 0;
        remaining -= num_colonies[i];
    }
    num_colonies[effective_imperialists - 1] = remaining;

    // Shuffle colony indices
    int colony_indices[NUM_COUNTRIES];
    for (int i = 0; i < num_colonies_total; i++) colony_indices[i] = effective_imperialists + i;
    for (int i = num_colonies_total - 1; i > 0; i--) {
        int j = (int)(rand_double(0, i + 1));
        int temp = colony_indices[i];
        colony_indices[i] = colony_indices[j];
        colony_indices[j] = temp;
    }

    // Initialize empires
    int colony_idx = 0;
    for (int i = 0; i < *num_empires; i++) {
        Empire *emp = &(*empires)[i];
        emp->num_colonies = num_colonies[i];
        emp->imperialist_cost = costs[indices[i]];
        emp->imperialist_position = (double *)malloc(opt->dim * sizeof(double));
        if (!emp->imperialist_position) {
            printf("Memory allocation failed for imperialist_position at empire %d\n", i);
            for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
            exit(1);
        }
        memcpy(emp->imperialist_position, opt->population[indices[i]].position, opt->dim * sizeof(double));

        if (emp->num_colonies > 0) {
            emp->colonies_position = (double **)malloc(emp->num_colonies * sizeof(double *));
            emp->colonies_cost = (double *)malloc(emp->num_colonies * sizeof(double));
            if (!emp->colonies_position || !emp->colonies_cost) {
                printf("Memory allocation failed for colonies at empire %d\n", i);
                free(emp->imperialist_position);
                free(emp->colonies_position);
                free(emp->colonies_cost);
                for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
                exit(1);
            }
            for (int j = 0; j < emp->num_colonies; j++) {
                emp->colonies_position[j] = (double *)malloc(opt->dim * sizeof(double));
                if (!emp->colonies_position[j]) {
                    printf("Memory allocation failed for colony position %d in empire %d\n", j, i);
                    for (int k = 0; k < j; k++) free(emp->colonies_position[k]);
                    free(emp->colonies_position);
                    free(emp->colonies_cost);
                    free(emp->imperialist_position);
                    for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
                    exit(1);
                }
                if (colony_idx >= num_colonies_total) {
                    printf("Colony index out of bounds: %d >= %d\n", colony_idx, num_colonies_total);
                    for (int k = 0; k < j; k++) free(emp->colonies_position[k]);
                    free(emp->colonies_position);
                    free(emp->colonies_cost);
                    free(emp->imperialist_position);
                    for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
                    exit(1);
                }
                memcpy(emp->colonies_position[j], opt->population[colony_indices[colony_idx]].position, opt->dim * sizeof(double));
                emp->colonies_cost[j] = costs[colony_indices[colony_idx]];
                colony_idx++;
            }
        } else {
            emp->num_colonies = 1;
            emp->colonies_position = (double **)malloc(sizeof(double *));
            emp->colonies_cost = (double *)malloc(sizeof(double));
            if (!emp->colonies_position || !emp->colonies_cost) {
                printf("Memory allocation failed for single colony at empire %d\n", i);
                free(emp->colonies_position);
                free(emp->colonies_cost);
                free(emp->imperialist_position);
                for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
                exit(1);
            }
            emp->colonies_position[0] = (double *)malloc(opt->dim * sizeof(double));
            if (!emp->colonies_position[0]) {
                printf("Memory allocation failed for single colony position at empire %d\n", i);
                free(emp->colonies_position);
                free(emp->colonies_cost);
                free(emp->imperialist_position);
                for (int k = 0; k < i; k++) free_empires((*empires), k + 1, opt->dim);
                exit(1);
            }
            for (int j = 0; j < opt->dim; j++) {
                emp->colonies_position[0][j] = opt->bounds[2 * j] + rand_double(0, 1) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
            }
            emp->colonies_cost[0] = params->objective_function(emp->colonies_position[0]);
            data->fes++;
        }

        double mean_colony_cost = 0.0;
        for (int j = 0; j < emp->num_colonies; j++) mean_colony_cost += emp->colonies_cost[j];
        mean_colony_cost /= emp->num_colonies;
        emp->total_cost = emp->imperialist_cost + ZETA * mean_colony_cost;
    }
}

// Assimilation
void assimilate_colonies(Optimizer *opt, Empire *empire) {
    if (!opt || !empire || empire->num_colonies <= 0 || !empire->colonies_position || !empire->imperialist_position) return;

    const double *restrict imperialist_pos = empire->imperialist_position;
    const double *restrict bounds = opt->bounds;
    for (int i = 0; i < empire->num_colonies; i++) {
        double *restrict colony_pos = empire->colonies_position[i];
        for (int j = 0; j < opt->dim; j++) {
            double dist = imperialist_pos[j] - colony_pos[j];
            colony_pos[j] += ASSIMILATION_COEFF * rand_double(0, 1) * dist;
            if (colony_pos[j] < bounds[2 * j]) colony_pos[j] = bounds[2 * j];
            if (colony_pos[j] > bounds[2 * j + 1]) colony_pos[j] = bounds[2 * j + 1];
        }
    }
}

// Revolution
void revolve_colonies(Optimizer *opt, Empire *empire, double revolution_rate) {
    if (!opt || !empire || empire->num_colonies <= 0 || !empire->colonies_position) return;

    int num_revolving = (int)(revolution_rate * empire->num_colonies);
    if (num_revolving <= 0) return;

    const double *restrict bounds = opt->bounds;
    for (int i = 0; i < num_revolving; i++) {
        int idx = (int)(rand_double(0, empire->num_colonies));
        double *colony_pos = empire->colonies_position[idx];
        for (int j = 0; j < opt->dim; j++) {
            colony_pos[j] = bounds[2 * j] + rand_double(0, 1) * (opt->bounds[2 * j + 1] - opt->bounds[2 * j]);
        }
    }
}

// Combined possession and best solution update
void possess_and_update_best(Optimizer *opt, ICAParams *params, Empire *empire, ICAOptimizerData *data) {
    if (!opt || !params || !empire || empire->num_colonies <= 0 || !empire->colonies_position || !empire->colonies_cost) return;

    // Update colony costs and find best colony
    int best_colony_idx = 0;
    double min_colony_cost = INFINITY;
    for (int i = 0; i < empire->num_colonies; i++) {
        double cost = params->objective_function(empire->colonies_position[i]);
        empire->colonies_cost[i] = cost;
        data->fes++;
        if (cost < min_colony_cost) {
            min_colony_cost = cost;
            best_colony_idx = i;
        }
        if (cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = cost;
            memcpy(opt->best_solution.position, empire->colonies_position[i], opt->dim * sizeof(double));
        }
    }

    // Update imperialist cost
    empire->imperialist_cost = params->objective_function(empire->imperialist_position);
    data->fes++;
    if (empire->imperialist_cost < opt->best_solution.fitness) {
        opt->best_solution.fitness = empire->imperialist_cost;
        memcpy(opt->best_solution.position, empire->imperialist_position, opt->dim * sizeof(double));
    }

    // Swap if best colony is better
    if (min_colony_cost < empire->imperialist_cost) {
        double *temp_pos = empire->imperialist_position;
        empire->imperialist_position = empire->colonies_position[best_colony_idx];
        empire->colonies_position[best_colony_idx] = temp_pos;
        double temp_cost = empire->imperialist_cost;
        empire->imperialist_cost = min_colony_cost;
        empire->colonies_cost[best_colony_idx] = temp_cost;
    }
}

// Unite similar empires
void unite_similar_empires(Optimizer *opt, Empire **empires, int *num_empires) {
    if (!opt || !empires || !num_empires || *num_empires <= 1) return;

    double threshold = UNITING_THRESHOLD * (opt->bounds[2 * opt->dim - 1] - opt->bounds[2 * opt->dim - 2]);
    int i = 0;
    while (i < *num_empires - 1) {
        int j = i + 1;
        while (j < *num_empires) {
            double distance = 0.0;
            const double *pos_i = (*empires)[i].imperialist_position;
            const double *pos_j = (*empires)[j].imperialist_position;
            for (int k = 0; k < opt->dim; k++) {
                double diff = pos_i[k] - pos_j[k];
                distance += diff * diff;
            }
            if (distance > threshold * threshold) {
                j++;
                continue;
            }

            int better_idx = (*empires)[i].imperialist_cost < (*empires)[j].imperialist_cost ? i : j;
            int worse_idx = better_idx == i ? j : i;
            Empire *better = &(*empires)[better_idx];
            Empire *worse = &(*empires)[worse_idx];

            int new_num_colonies = better->num_colonies + worse->num_colonies + 1;
            double **new_positions = (double **)malloc(new_num_colonies * sizeof(double *));
            double *new_costs = (double *)malloc(new_num_colonies * sizeof(double));
            if (!new_positions || !new_costs) {
                free(new_positions);
                free(new_costs);
                return;
            }

            // Allocate new memory for all positions to avoid double free
            for (int k = 0; k < new_num_colonies; k++) {
                new_positions[k] = (double *)malloc(opt->dim * sizeof(double));
                if (!new_positions[k]) {
                    for (int m = 0; m < k; m++) free(new_positions[m]);
                    free(new_positions);
                    free(new_costs);
                    return;
                }
            }

            // Copy positions and costs
            for (int k = 0; k < better->num_colonies; k++) {
                memcpy(new_positions[k], better->colonies_position[k], opt->dim * sizeof(double));
                new_costs[k] = better->colonies_cost[k];
            }
            memcpy(new_positions[better->num_colonies], worse->imperialist_position, opt->dim * sizeof(double));
            new_costs[better->num_colonies] = worse->imperialist_cost;
            for (int k = 0; k < worse->num_colonies; k++) {
                memcpy(new_positions[better->num_colonies + 1 + k], worse->colonies_position[k], opt->dim * sizeof(double));
                new_costs[better->num_colonies + 1 + k] = worse->colonies_cost[k];
            }

            // Free old memory
            free(better->colonies_position);
            free(better->colonies_cost);
            better->colonies_position = new_positions;
            better->colonies_cost = new_costs;
            better->num_colonies = new_num_colonies;

            double mean_colony_cost = 0.0;
            for (int k = 0; k < better->num_colonies; k++) mean_colony_cost += better->colonies_cost[k];
            mean_colony_cost /= better->num_colonies;
            better->total_cost = better->imperialist_cost + ZETA * mean_colony_cost;

            // Clear worse empire's pointers to prevent double free
            worse->imperialist_position = NULL;
            worse->colonies_position = NULL;
            worse->colonies_cost = NULL;

            for (int k = worse_idx; k < *num_empires - 1; k++) (*empires)[k] = (*empires)[k + 1];
            (*num_empires)--;
            if (*num_empires <= 0) return;
        }
        i++;
    }

    if (*num_empires > 0) {
        Empire *temp = (Empire *)realloc(*empires, *num_empires * sizeof(Empire));
        if (temp) *empires = temp;
    }
}

// Imperialistic competition
void imperialistic_competition(Optimizer *opt, Empire **empires, int *num_empires) {
    if (!opt || !empires || !num_empires || *num_empires <= 1 || rand_double(0, 1) > 0.11) return;

    double total_costs[NUM_IMPERIALISTS];
    double powers[NUM_IMPERIALISTS];

    // Find weakest empire
    int weakest_idx = 0;
    double max_cost = (*empires)[0].total_cost;
    total_costs[0] = max_cost;
    for (int i = 1; i < *num_empires; i++) {
        total_costs[i] = (*empires)[i].total_cost;
        if (total_costs[i] > max_cost) {
            max_cost = total_costs[i];
            weakest_idx = i;
        }
    }

    // Compute power
    double power_sum = 0.0;
    for (int i = 0; i < *num_empires; i++) {
        powers[i] = max_cost - total_costs[i];
        power_sum += powers[i];
    }
    if (power_sum <= 0) return;

    // Select empire
    double max_diff = (powers[0] / power_sum) - rand_double(0, 1);
    int selected_idx = 0;
    for (int i = 1; i < *num_empires; i++) {
        double diff = (powers[i] / power_sum) - rand_double(0, 1);
        if (diff > max_diff) {
            max_diff = diff;
            selected_idx = i;
        }
    }

    if (selected_idx == weakest_idx || (*empires)[weakest_idx].num_colonies <= 0) return;

    Empire *selected = &(*empires)[selected_idx];
    Empire *weakest = &(*empires)[weakest_idx];
    int colony_idx = (int)(rand_double(0, weakest->num_colonies));

    // Transfer colony
    int new_selected_colonies = selected->num_colonies + 1;
    double **new_positions = (double **)malloc(new_selected_colonies * sizeof(double *));
    double *new_costs = (double *)malloc(new_selected_colonies * sizeof(double));
    if (!new_positions || !new_costs) {
        free(new_positions);
        free(new_costs);
        return;
    }
    for (int i = 0; i < new_selected_colonies; i++) {
        new_positions[i] = (double *)malloc(opt->dim * sizeof(double));
        if (!new_positions[i]) {
            for (int k = 0; k < i; k++) free(new_positions[k]);
            free(new_positions);
            free(new_costs);
            return;
        }
    }
    for (int i = 0; i < selected->num_colonies; i++) {
        memcpy(new_positions[i], selected->colonies_position[i], opt->dim * sizeof(double));
        new_costs[i] = selected->colonies_cost[i];
    }
    memcpy(new_positions[selected->num_colonies], weakest->colonies_position[colony_idx], opt->dim * sizeof(double));
    new_costs[selected->num_colonies] = weakest->colonies_cost[colony_idx];

    int new_weakest_colonies = weakest->num_colonies - 1;
    double **new_weakest_positions = NULL;
    double *new_weakest_costs = NULL;
    if (new_weakest_colonies > 0) {
        new_weakest_positions = (double **)malloc(new_weakest_colonies * sizeof(double *));
        new_weakest_costs = (double *)malloc(new_weakest_colonies * sizeof(double));
        if (!new_weakest_positions || !new_weakest_costs) {
            for (int k = 0; k < new_selected_colonies; k++) free(new_positions[k]);
            free(new_positions);
            free(new_costs);
            free(new_weakest_positions);
            free(new_weakest_costs);
            return;
        }
        for (int i = 0; i < new_weakest_colonies; i++) {
            new_weakest_positions[i] = (double *)malloc(opt->dim * sizeof(double));
            if (!new_weakest_positions[i]) {
                for (int k = 0; k < i; k++) free(new_weakest_positions[k]);
                for (int k = 0; k < new_selected_colonies; k++) free(new_positions[k]);
                free(new_positions);
                free(new_costs);
                free(new_weakest_positions);
                free(new_weakest_costs);
                return;
            }
        }
        for (int i = 0, k = 0; i < weakest->num_colonies; i++) {
            if (i != colony_idx) {
                memcpy(new_weakest_positions[k], weakest->colonies_position[i], opt->dim * sizeof(double));
                new_weakest_costs[k] = weakest->colonies_cost[i];
                k++;
            }
        }
    }

    free(selected->colonies_position);
    free(selected->colonies_cost);
    selected->colonies_position = new_positions;
    selected->colonies_cost = new_costs;
    selected->num_colonies = new_selected_colonies;

    free(weakest->colonies_position);
    free(weakest->colonies_cost);
    weakest->colonies_position = new_weakest_positions;
    weakest->colonies_cost = new_weakest_costs;
    weakest->num_colonies = new_weakest_colonies;

    // Handle empire collapse
    if (weakest->num_colonies <= 0) {
        new_selected_colonies = selected->num_colonies + 1;
        new_positions = (double **)malloc(new_selected_colonies * sizeof(double *));
        new_costs = (double *)malloc(new_selected_colonies * sizeof(double));
        if (!new_positions || !new_costs) {
            free(new_positions);
            free(new_costs);
            return;
        }
        for (int i = 0; i < new_selected_colonies; i++) {
            new_positions[i] = (double *)malloc(opt->dim * sizeof(double));
            if (!new_positions[i]) {
                for (int k = 0; k < i; k++) free(new_positions[k]);
                free(new_positions);
                free(new_costs);
                return;
            }
        }
        for (int i = 0; i < selected->num_colonies; i++) {
            memcpy(new_positions[i], selected->colonies_position[i], opt->dim * sizeof(double));
            new_costs[i] = selected->colonies_cost[i];
        }
        memcpy(new_positions[selected->num_colonies], weakest->imperialist_position, opt->dim * sizeof(double));
        new_costs[selected->num_colonies] = weakest->imperialist_cost;

        free(selected->colonies_position);
        free(selected->colonies_cost);
        selected->colonies_position = new_positions;
        selected->colonies_cost = new_costs;
        selected->num_colonies = new_selected_colonies;

        free(weakest->imperialist_position);
        weakest->imperialist_position = NULL;
        for (int k = weakest_idx; k < *num_empires - 1; k++) (*empires)[k] = (*empires)[k + 1];
        (*num_empires)--;
        if (*num_empires <= 0) return;

        Empire *temp = (Empire *)realloc(*empires, *num_empires * sizeof(Empire));
        if (temp) *empires = temp;
    }

    // Update total costs
    double mean_colony_cost = 0.0;
    for (int j = 0; j < selected->num_colonies; j++) mean_colony_cost += selected->colonies_cost[j];
    mean_colony_cost /= selected->num_colonies;
    selected->total_cost = selected->imperialist_cost + ZETA * mean_colony_cost;

    if (weakest->num_colonies > 0) {
        mean_colony_cost = 0.0;
        for (int j = 0; j < weakest->num_colonies; j++) mean_colony_cost += weakest->colonies_cost[j];
        mean_colony_cost /= weakest->num_colonies;
        weakest->total_cost = weakest->imperialist_cost + ZETA * mean_colony_cost;
    }
}

// Free empire memory
void free_empires(Empire *empires, int num_empires, int dim) {
    if (!empires) return;
    for (int i = 0; i < num_empires; i++) {
        if (empires[i].imperialist_position) {
            free(empires[i].imperialist_position);
            empires[i].imperialist_position = NULL;
        }
        if (empires[i].colonies_position) {
            for (int j = 0; j < empires[i].num_colonies; j++) {
                if (empires[i].colonies_position[j]) {
                    free(empires[i].colonies_position[j]);
                    empires[i].colonies_position[j] = NULL;
                }
            }
            free(empires[i].colonies_position);
            empires[i].colonies_position = NULL;
        }
        if (empires[i].colonies_cost) {
            free(empires[i].colonies_cost);
            empires[i].colonies_cost = NULL;
        }
    }
    free(empires);
}

// Main optimization function
void ICA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function || !opt->population || !opt->bounds) {
        printf("Invalid arguments in ICA_optimize\n");
        return;
    }
    if (opt->dim <= 0 || opt->population_size <= 0 || opt->max_iter <= 0) {
        printf("Invalid optimizer parameters: dim=%d, population_size=%d, max_iter=%d\n",
               opt->dim, opt->population_size, opt->max_iter);
        return;
    }

    srand((unsigned)time(NULL));
    ICAParams params = {
        .objective_function = objective_function,
        .stop_if_single_empire = 1,
        .revolution_rate = REVOLUTION_RATE
    };
    ICAOptimizerData data = { .fes = 0 };

    // Initialize best solution
    opt->best_solution.fitness = INFINITY;
    if (!opt->best_solution.position) {
        opt->best_solution.position = (double *)malloc(opt->dim * sizeof(double));
        if (!opt->best_solution.position) {
            printf("Memory allocation failed for best_solution.position\n");
            return;
        }
    }

    Empire *empires = NULL;
    int num_empires = 0;
    create_initial_empires(opt, &params, &empires, &num_empires, &data);
    if (!empires) {
        printf("Failed to create initial empires\n");
        free(opt->best_solution.position);
        opt->best_solution.position = NULL;
        return;
    }

    char output_buffer[256];
    int output_len = 0;

    for (int decade = 0; decade < MAX_DECADES && num_empires > 0; decade++) {
        params.revolution_rate *= DAMP_RATIO;
        for (int i = 0; i < num_empires; i++) {
            assimilate_colonies(opt, &empires[i]);
            revolve_colonies(opt, &empires[i], params.revolution_rate);
            possess_and_update_best(opt, &params, &empires[i], &data);
            double mean_colony_cost = 0.0;
            for (int j = 0; j < empires[i].num_colonies; j++) mean_colony_cost += empires[i].colonies_cost[j];
            mean_colony_cost /= empires[i].num_colonies;
            empires[i].total_cost = empires[i].imperialist_cost + ZETA * mean_colony_cost;
        }

        unite_similar_empires(opt, &empires, &num_empires);
        imperialistic_competition(opt, &empires, &num_empires);

        output_len += snprintf(output_buffer + output_len, sizeof(output_buffer) - output_len,
                               "Decade %d: Empires = %d, Best Cost = %f, FES = %d\n",
                               decade + 1, num_empires, opt->best_solution.fitness, data.fes);
        if (output_len >= sizeof(output_buffer) - 100) {
            printf("%s", output_buffer);
            output_len = 0;
        }

        if (num_empires == 1 && params.stop_if_single_empire) {
            output_len += snprintf(output_buffer + output_len, sizeof(output_buffer) - output_len,
                                   "Single empire remaining, stopping\n");
            break;
        }
    }

    if (output_len > 0) printf("%s", output_buffer);
    printf("ICA optimization completed with %d function evaluations\n", data.fes);

    free_empires(empires, num_empires, opt->dim);
}

// Wrapper
void ICA_optimize_wrapper(void *opt_void, double (*objective_function)(double *)) {
    Optimizer *opt = (Optimizer *)opt_void;
    if (!opt) {
        printf("Invalid optimizer in ICA_optimize_wrapper\n");
        return;
    }
    ICA_optimize(opt, objective_function);
}
