#include "TFWO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate a random double between min and max
double rand_double(double min, double max);

// Enforce bounds on a position
void enforce_bounds(double *position, double *bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        position[i] = fmax(bounds[2 * i], fmin(bounds[2 * i + 1], position[i]));
    }
}

// Initialize whirlpools and objects
void initialize_whirlpools(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in initialize_whirlpools\n");
        return;
    }

    int i, j, k;
    data->whirlpools = (TFWO_Whirlpool *)calloc(data->n_whirlpools, sizeof(TFWO_Whirlpool));
    if (!data->whirlpools) {
        fprintf(stderr, "Error: Failed to allocate whirlpools\n");
        return;
    }

    // Temporary array to store all objects
    typedef struct {
        double *position;
        double cost;
        double delta;
    } TempObject;
    int n_pop = data->n_whirlpools + data->n_whirlpools * data->n_objects_per_whirlpool;
    TempObject *objects = (TempObject *)calloc(n_pop, sizeof(TempObject));
    if (!objects) {
        fprintf(stderr, "Error: Failed to allocate temporary objects\n");
        free(data->whirlpools);
        return;
    }

    // Initialize objects
    for (i = 0; i < n_pop; i++) {
        objects[i].position = (double *)calloc(opt->dim, sizeof(double));
        if (!objects[i].position) {
            fprintf(stderr, "Error: Failed to allocate object position %d\n", i);
            for (j = 0; j < i; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
        objects[i].delta = 0.0;
        for (j = 0; j < opt->dim; j++) {
            objects[i].position[j] = rand_double(opt->bounds[2 * j], opt->bounds[2 * j + 1]);
        }
        objects[i].cost = objective_function(objects[i].position);
    }

    // Sort objects by cost
    for (i = 0; i < n_pop - 1; i++) {
        for (j = 0; j < n_pop - i - 1; j++) {
            if (objects[j].cost > objects[j + 1].cost) {
                TempObject temp = objects[j];
                objects[j] = objects[j + 1];
                objects[j + 1] = temp;
            }
        }
    }

    // Initialize whirlpools
    for (i = 0; i < data->n_whirlpools; i++) {
        data->whirlpools[i].position = (double *)calloc(opt->dim, sizeof(double));
        if (!data->whirlpools[i].position) {
            fprintf(stderr, "Error: Failed to allocate whirlpool position %d\n", i);
            for (j = 0; j < i; j++) {
                free(data->whirlpools[j].position);
                for (k = 0; k < data->whirlpools[j].n_objects; k++) {
                    free(data->whirlpools[j].objects[k].position);
                }
                free(data->whirlpools[j].objects);
            }
            for (j = 0; j < n_pop; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
        memcpy(data->whirlpools[i].position, objects[i].position, opt->dim * sizeof(double));
        data->whirlpools[i].cost = objects[i].cost;
        data->whirlpools[i].delta = objects[i].delta;
        data->whirlpools[i].n_objects = data->n_objects_per_whirlpool;
        data->whirlpools[i].objects = (TFWO_Object *)calloc(data->n_objects_per_whirlpool, sizeof(TFWO_Object));
        if (!data->whirlpools[i].objects) {
            fprintf(stderr, "Error: Failed to allocate objects for whirlpool %d\n", i);
            for (j = 0; j <= i; j++) free(data->whirlpools[j].position);
            for (j = 0; j < n_pop; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
    }

    // Distribute remaining objects to whirlpools
    int obj_idx = data->n_whirlpools;
    for (i = 0; i < data->n_whirlpools; i++) {
        for (j = 0; j < data->n_objects_per_whirlpool; j++) {
            data->whirlpools[i].objects[j].position = (double *)calloc(opt->dim, sizeof(double));
            if (!data->whirlpools[i].objects[j].position) {
                fprintf(stderr, "Error: Failed to allocate object position %d in whirlpool %d\n", j, i);
                for (k = 0; k <= i; k++) {
                    for (int m = 0; m < (k == i ? j : data->n_objects_per_whirlpool); m++) {
                        free(data->whirlpools[k].objects[m].position);
                    }
                    free(data->whirlpools[k].objects);
                    free(data->whirlpools[k].position);
                }
                for (j = 0; j < n_pop; j++) free(objects[j].position);
                free(objects);
                free(data->whirlpools);
                return;
            }
            memcpy(data->whirlpools[i].objects[j].position, objects[obj_idx].position, opt->dim * sizeof(double));
            data->whirlpools[i].objects[j].cost = objects[obj_idx].cost;
            data->whirlpools[i].objects[j].delta = objects[obj_idx].delta;
            obj_idx++;
        }
    }

    // Free temporary objects
    for (i = 0; i < n_pop; i++) {
        free(objects[i].position);
    }
    free(objects);
}

// Update objects and whirlpool positions
void effects_of_whirlpools(Optimizer *opt, TFWO_Data *data, int iter, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in effects_of_whirlpools\n");
        return;
    }

    int i, j, k, t;
    double *d = (double *)calloc(opt->dim, sizeof(double));
    double *d2 = (double *)calloc(opt->dim, sizeof(double));
    double *RR = (double *)calloc(opt->dim, sizeof(double));
    if (!d || !d2 || !RR) {
        fprintf(stderr, "Error: Failed to allocate temporary arrays in effects_of_whirlpools\n");
        free(d); free(d2); free(RR);
        return;
    }

    for (i = 0; i < data->n_whirlpools; i++) {
        for (j = 0; j < data->whirlpools[i].n_objects; j++) {
            int min_idx = i, max_idx = i;
            // Compute influence from other whirlpools
            if (data->n_whirlpools > 1) {
                double *J = (double *)calloc(data->n_whirlpools - 1, sizeof(double));
                if (!J) {
                    fprintf(stderr, "Error: Failed to allocate J array\n");
                    free(d); free(d2); free(RR); free(J);
                    return;
                }
                int J_idx = 0;
                double sum_t, sum_obj;
                for (t = 0; t < data->n_whirlpools; t++) {
                    if (t != i) {
                        sum_t = sum_obj = 0.0;
                        for (k = 0; k < opt->dim; k++) {
                            sum_t += data->whirlpools[t].position[k];
                            sum_obj += data->whirlpools[i].objects[j].position[k];
                        }
                        J[J_idx] = pow(fabs(data->whirlpools[t].cost), 1.0) * sqrt(fabs(sum_t - sum_obj));
                        J_idx++;
                    }
                }
                min_idx = 0;
                max_idx = 0;
                for (t = 1; t < data->n_whirlpools - 1; t++) {
                    if (J[t] < J[min_idx]) min_idx = t;
                    if (J[t] > J[max_idx]) max_idx = t;
                }
                if (min_idx >= i) min_idx++;
                if (max_idx >= i) max_idx++;
                for (k = 0; k < opt->dim; k++) {
                    d[k] = rand_double(0.0, 1.0) * (data->whirlpools[min_idx].position[k] - data->whirlpools[i].objects[j].position[k]);
                    d2[k] = rand_double(0.0, 1.0) * (data->whirlpools[max_idx].position[k] - data->whirlpools[i].objects[j].position[k]);
                }
                free(J);
            } else {
                for (k = 0; k < opt->dim; k++) {
                    d[k] = rand_double(0.0, 1.0) * (data->whirlpools[i].position[k] - data->whirlpools[i].objects[j].position[k]);
                    d2[k] = 0.0;
                }
            }

            // Update delta
            data->whirlpools[i].objects[j].delta += rand_double(0.0, 1.0) * rand_double(0.0, 1.0) * PI;
            double eee = data->whirlpools[i].objects[j].delta;
            double fr0 = cos(eee);
            double fr10 = -sin(eee);

            // Compute new position
            for (k = 0; k < opt->dim; k++) {
                double x = ((fr0 * d[k]) + (fr10 * d2[k])) * (1.0 + fabs(fr0 * fr10));
                RR[k] = data->whirlpools[i].position[k] - x;
            }
            enforce_bounds(RR, opt->bounds, opt->dim);
            double cost = objective_function(RR);

            // Update if better
            if (cost <= data->whirlpools[i].objects[j].cost) {
                data->whirlpools[i].objects[j].cost = cost;
                memcpy(data->whirlpools[i].objects[j].position, RR, opt->dim * sizeof(double));
            }

            // Random jump
            double FE_i = pow(fabs(pow(cos(eee), 2) * pow(sin(eee), 2)), 2);
            if (rand_double(0.0, 1.0) < FE_i) {
                k = rand() % opt->dim;
                data->whirlpools[i].objects[j].position[k] = rand_double(opt->bounds[2 * k], opt->bounds[2 * k + 1]);
                data->whirlpools[i].objects[j].cost = objective_function(data->whirlpools[i].objects[j].position);
            }
        }

        // Update whirlpool position
        double *J = (double *)calloc(data->n_whirlpools, sizeof(double));
        if (!J) {
            fprintf(stderr, "Error: Failed to allocate J array for whirlpool update\n");
            free(d); free(d2); free(RR); free(J);
            return;
        }
        double sum_t, sum_i;
        for (t = 0; t < data->n_whirlpools; t++) {
            sum_t = sum_i = 0.0;
            for (k = 0; k < opt->dim; k++) {
                sum_t += data->whirlpools[t].position[k];
                sum_i += data->whirlpools[i].position[k];
            }
            J[t] = (t == i) ? INFINITY : data->whirlpools[t].cost * fabs(sum_t - sum_i);
        }
        int min_idx = 0;
        for (t = 1; t < data->n_whirlpools; t++) {
            if (J[t] < J[min_idx]) min_idx = t;
        }
        data->whirlpools[i].delta += rand_double(0.0, 1.0) * rand_double(0.0, 1.0) * PI;
        double fr = fabs(cos(data->whirlpools[i].delta) + sin(data->whirlpools[i].delta));
        double *new_position = (double *)calloc(opt->dim, sizeof(double));
        if (!new_position) {
            fprintf(stderr, "Error: Failed to allocate new_position\n");
            free(d); free(d2); free(RR); free(J); free(new_position);
            return;
        }
        for (k = 0; k < opt->dim; k++) {
            double x = fr * rand_double(0.0, 1.0) * (data->whirlpools[min_idx].position[k] - data->whirlpools[i].position[k]);
            new_position[k] = data->whirlpools[min_idx].position[k] - x;
        }
        enforce_bounds(new_position, opt->bounds, opt->dim);
        double new_cost = objective_function(new_position);
        if (new_cost <= data->whirlpools[i].cost) {
            data->whirlpools[i].cost = new_cost;
            memcpy(data->whirlpools[i].position, new_position, opt->dim * sizeof(double));
        }
        free(new_position);
        free(J);
    }

    free(d);
    free(d2);
    free(RR);
}

// Update whirlpool with best object if better
void update_best_whirlpool(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in update_best_whirlpool\n");
        return;
    }

    int i, j;
    for (i = 0; i < data->n_whirlpools; i++) {
        double min_cost = data->whirlpools[i].objects[0].cost;
        int min_cost_idx = 0;
        for (j = 1; j < data->whirlpools[i].n_objects; j++) {
            if (data->whirlpools[i].objects[j].cost < min_cost) {
                min_cost = data->whirlpools[i].objects[j].cost;
                min_cost_idx = j;
            }
        }
        if (min_cost <= data->whirlpools[i].cost) {
            double *temp_pos = (double *)calloc(opt->dim, sizeof(double));
            if (!temp_pos) {
                fprintf(stderr, "Error: Failed to allocate temp_pos in update_best_whirlpool\n");
                return;
            }
            memcpy(temp_pos, data->whirlpools[i].position, opt->dim * sizeof(double));
            double temp_cost = data->whirlpools[i].cost;
            memcpy(data->whirlpools[i].position, data->whirlpools[i].objects[min_cost_idx].position, opt->dim * sizeof(double));
            data->whirlpools[i].cost = min_cost;
            memcpy(data->whirlpools[i].objects[min_cost_idx].position, temp_pos, opt->dim * sizeof(double));
            data->whirlpools[i].objects[min_cost_idx].cost = temp_cost;
            free(temp_pos);
        }
    }
}

// Free TFWO data
void free_tfwo_data(TFWO_Data *data) {
    if (!data) return;
    if (data->whirlpools) {
        for (int i = 0; i < data->n_whirlpools; i++) {
            if (data->whirlpools[i].objects) {
                for (int j = 0; j < data->whirlpools[i].n_objects; j++) {
                    free(data->whirlpools[i].objects[j].position);
                }
                free(data->whirlpools[i].objects);
            }
            free(data->whirlpools[i].position);
        }
        free(data->whirlpools);
    }
    free(data->best_costs);
    free(data->mean_costs);
    free(data);
}

// Main optimization function
void TFWO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Null pointer in TFWO_optimize\n");
        return;
    }

    srand((unsigned int)time(NULL));

    TFWO_Data *data = (TFWO_Data *)calloc(1, sizeof(TFWO_Data));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate TFWO_Data\n");
        return;
    }
    data->n_whirlpools = N_WHIRLPOOLS_DEFAULT;
    data->n_objects_per_whirlpool = N_OBJECTS_PER_WHIRLPOOL_DEFAULT;
    data->best_costs = (double *)calloc(opt->max_iter, sizeof(double));
    data->mean_costs = (double *)calloc(opt->max_iter, sizeof(double));
    if (!data->best_costs || !data->mean_costs) {
        fprintf(stderr, "Error: Failed to allocate cost arrays\n");
        free(data->best_costs);
        free(data->mean_costs);
        free(data);
        return;
    }

    initialize_whirlpools(opt, data, objective_function);
    if (!data->whirlpools) {
        fprintf(stderr, "Error: Whirlpool initialization failed\n");
        free_tfwo_data(data);
        return;
    }

    for (int iter = 0; iter < opt->max_iter; iter++) {
        effects_of_whirlpools(opt, data, iter, objective_function);
        update_best_whirlpool(opt, data, objective_function);

        // Find best whirlpool
        double best_cost = data->whirlpools[0].cost;
        int best_idx = 0;
        double mean_cost = data->whirlpools[0].cost;
        for (int i = 1; i < data->n_whirlpools; i++) {
            if (data->whirlpools[i].cost < best_cost) {
                best_cost = data->whirlpools[i].cost;
                best_idx = i;
            }
            mean_cost += data->whirlpools[i].cost;
        }
        mean_cost /= data->n_whirlpools;

        // Update global best
        if (best_cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_cost;
            memcpy(opt->best_solution.position, data->whirlpools[best_idx].position, opt->dim * sizeof(double));
        }

        // Store iteration results
        data->best_costs[iter] = best_cost;
        data->mean_costs[iter] = mean_cost;

        printf("Iter %d: Best Cost = %f\n", iter + 1, best_cost);
    }

    free_tfwo_data(data);
}
