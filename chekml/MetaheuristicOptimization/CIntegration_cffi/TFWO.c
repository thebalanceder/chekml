#include "TFWO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Generate a random double between min and max
inline double rand_double_tfwo(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Enforce bounds on a position
inline void enforce_bounds(double *position, double *bounds, int dim) {
    double *pos = position, *b = bounds;
    for (int i = 0; i < dim; i++, pos++, b += 2) {
        *pos = fmax(*b, fmin(*(b + 1), *pos));
    }
}

// Comparison function for qsort
static int compare_objects(const void *a, const void *b) {
    double cost_a = ((TFWO_Object *)a)->cost;
    double cost_b = ((TFWO_Object *)b)->cost;
    return (cost_a > cost_b) - (cost_a < cost_b);
}

// Initialize whirlpools and objects
void initialize_whirlpools(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in initialize_whirlpools\n");
        return;
    }

    int dim = opt->dim;
    int n_pop = data->n_whirlpools + data->n_whirlpools * data->n_objects_per_whirlpool;
    data->whirlpools = (TFWO_Whirlpool *)calloc(data->n_whirlpools, sizeof(TFWO_Whirlpool));
    if (!data->whirlpools) {
        fprintf(stderr, "Error: Failed to allocate whirlpools\n");
        return;
    }

    // Allocate temporary objects
    TFWO_Object *objects = (TFWO_Object *)calloc(n_pop, sizeof(TFWO_Object));
    if (!objects) {
        fprintf(stderr, "Error: Failed to allocate temporary objects\n");
        free(data->whirlpools);
        return;
    }

    // Initialize objects
    for (int i = 0; i < n_pop; i++) {
        objects[i].position = (double *)calloc(dim, sizeof(double));
        if (!objects[i].position) {
            fprintf(stderr, "Error: Failed to allocate object position %d\n", i);
            for (int j = 0; j < i; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
        objects[i].delta = 0.0;
        double *pos = objects[i].position;
        double *b = opt->bounds;
        for (int j = 0; j < dim; j++, b += 2) {
            pos[j] = rand_double_tfwo(*b, *(b + 1));
        }
        objects[i].cost = objective_function(pos);
    }

    // Sort objects by cost using qsort
    qsort(objects, n_pop, sizeof(TFWO_Object), compare_objects);

    // Initialize whirlpools
    for (int i = 0; i < data->n_whirlpools; i++) {
        data->whirlpools[i].position = (double *)calloc(dim, sizeof(double));
        if (!data->whirlpools[i].position) {
            fprintf(stderr, "Error: Failed to allocate whirlpool position %d\n", i);
            for (int j = 0; j < i; j++) {
                free(data->whirlpools[j].position);
                for (int k = 0; k < data->whirlpools[j].n_objects; k++) {
                    free(data->whirlpools[j].objects[k].position);
                }
                free(data->whirlpools[j].objects);
            }
            for (int j = 0; j < n_pop; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
        double *pos = data->whirlpools[i].position;
        double *src = objects[i].position;
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
            pos[j] = src[j];
            sum += pos[j];
        }
        data->whirlpools[i].position_sum = sum;
        data->whirlpools[i].cost = objects[i].cost;
        data->whirlpools[i].delta = objects[i].delta;
        data->whirlpools[i].n_objects = data->n_objects_per_whirlpool;
        data->whirlpools[i].objects = (TFWO_Object *)calloc(data->n_objects_per_whirlpool, sizeof(TFWO_Object));
        if (!data->whirlpools[i].objects) {
            fprintf(stderr, "Error: Failed to allocate objects for whirlpool %d\n", i);
            for (int j = 0; j <= i; j++) free(data->whirlpools[j].position);
            for (int j = 0; j < n_pop; j++) free(objects[j].position);
            free(objects);
            free(data->whirlpools);
            return;
        }
    }

    // Distribute remaining objects to whirlpools
    int obj_idx = data->n_whirlpools;
    for (int i = 0; i < data->n_whirlpools; i++) {
        TFWO_Object *objs = data->whirlpools[i].objects;
        for (int j = 0; j < data->n_objects_per_whirlpool; j++, obj_idx++) {
            objs[j].position = objects[obj_idx].position; // Transfer ownership
            objects[obj_idx].position = NULL; // Prevent double-free
            objs[j].cost = objects[obj_idx].cost;
            objs[j].delta = objects[obj_idx].delta;
        }
    }

    // Free remaining temporary objects
    for (int i = 0; i < n_pop; i++) {
        free(objects[i].position); // Safe for NULL
    }
    free(objects);
}

// Update objects and whirlpool positions
void effects_of_whirlpools(Optimizer *opt, TFWO_Data *data, int iter, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in effects_of_whirlpools\n");
        return;
    }

    int dim = opt->dim;
    double *d = data->temp_d;
    double *d2 = data->temp_d2;
    double *RR = data->temp_RR;
    double *J = data->temp_J;

    for (int i = 0; i < data->n_whirlpools; i++) {
        TFWO_Whirlpool *wp = &data->whirlpools[i];
        for (int j = 0; j < wp->n_objects; j++) {
            TFWO_Object *obj = &wp->objects[j];
            int min_idx = i, max_idx = i;

            // Compute influence from other whirlpools
            if (data->n_whirlpools > 1) {
                int J_idx = 0;
                double sum_obj = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum_obj += obj->position[k];
                }
                for (int t = 0; t < data->n_whirlpools; t++) {
                    if (t != i) {
                        J[J_idx] = fabs(data->whirlpools[t].cost) * sqrt(fabs(data->whirlpools[t].position_sum - sum_obj));
                        J_idx++;
                    }
                }
                min_idx = 0;
                max_idx = 0;
                for (int t = 1; t < data->n_whirlpools - 1; t++) {
                    if (J[t] < J[min_idx]) min_idx = t;
                    if (J[t] > J[max_idx]) max_idx = t;
                }
                if (min_idx >= i) min_idx++;
                if (max_idx >= i) max_idx++;
                double *min_pos = data->whirlpools[min_idx].position;
                double *max_pos = data->whirlpools[max_idx].position;
                double *obj_pos = obj->position;
                for (int k = 0; k < dim; k++) {
                    d[k] = rand_double_tfwo(0.0, 1.0) * (min_pos[k] - obj_pos[k]);
                    d2[k] = rand_double_tfwo(0.0, 1.0) * (max_pos[k] - obj_pos[k]);
                }
            } else {
                double *wp_pos = wp->position;
                double *obj_pos = obj->position;
                for (int k = 0; k < dim; k++) {
                    d[k] = rand_double_tfwo(0.0, 1.0) * (wp_pos[k] - obj_pos[k]);
                    d2[k] = 0.0;
                }
            }

            // Update delta and compute trigonometrics once
            obj->delta += rand_double_tfwo(0.0, 1.0) * rand_double_tfwo(0.0, 1.0) * PI;
            double eee = obj->delta;
            double cos_eee = cos(eee);
            double sin_eee = sin(eee);
            double fr0 = cos_eee;
            double fr10 = -sin_eee;
            double fr0_fr10 = fabs(fr0 * fr10);

            // Compute new position
            double *wp_pos = wp->position;
            for (int k = 0; k < dim; k++) {
                double x = (fr0 * d[k] + fr10 * d2[k]) * (1.0 + fr0_fr10);
                RR[k] = wp_pos[k] - x;
            }
            enforce_bounds(RR, opt->bounds, dim);
            double cost = objective_function(RR);

            // Update if better
            if (cost <= obj->cost) {
                obj->cost = cost;
                double *obj_pos = obj->position;
                for (int k = 0; k < dim; k++) {
                    obj_pos[k] = RR[k];
                }
            }

            // Random jump
            double cos_eee_sq = cos_eee * cos_eee;
            double sin_eee_sq = sin_eee * sin_eee;
            double FE_i = (cos_eee_sq * sin_eee_sq) * (cos_eee_sq * sin_eee_sq);
            if (rand_double_tfwo(0.0, 1.0) < FE_i) {
                int k = rand() % dim;
                obj->position[k] = rand_double_tfwo(opt->bounds[2 * k], opt->bounds[2 * k + 1]);
                obj->cost = objective_function(obj->position);
            }
        }

        // Update whirlpool position
        double sum_i = wp->position_sum;
        for (int t = 0; t < data->n_whirlpools; t++) {
            J[t] = (t == i) ? INFINITY : data->whirlpools[t].cost * fabs(data->whirlpools[t].position_sum - sum_i);
        }
        int min_idx = 0;
        for (int t = 1; t < data->n_whirlpools; t++) {
            if (J[t] < J[min_idx]) min_idx = t;
        }
        wp->delta += rand_double_tfwo(0.0, 1.0) * rand_double_tfwo(0.0, 1.0) * PI;
        double fr = fabs(cos(wp->delta) + sin(wp->delta));
        double *new_position = RR; // Reuse RR to avoid allocation
        double *min_pos = data->whirlpools[min_idx].position;
        double sum = 0.0;
        for (int k = 0; k < dim; k++) {
            double x = fr * rand_double_tfwo(0.0, 1.0) * (min_pos[k] - wp->position[k]);
            new_position[k] = min_pos[k] - x;
            sum += new_position[k];
        }
        enforce_bounds(new_position, opt->bounds, dim);
        double new_cost = objective_function(new_position);
        if (new_cost <= wp->cost) {
            wp->cost = new_cost;
            double *pos = wp->position;
            for (int k = 0; k < dim; k++) {
                pos[k] = new_position[k];
            }
            wp->position_sum = sum;
        }
    }
}

// Update whirlpool with best object if better
void update_best_whirlpool(Optimizer *opt, TFWO_Data *data, double (*objective_function)(double *)) {
    if (!opt || !data || !objective_function) {
        fprintf(stderr, "Error: Null pointer in update_best_whirlpool\n");
        return;
    }

    int dim = opt->dim;
    for (int i = 0; i < data->n_whirlpools; i++) {
        TFWO_Whirlpool *wp = &data->whirlpools[i];
        double min_cost = wp->objects[0].cost;
        int min_cost_idx = 0;
        for (int j = 1; j < wp->n_objects; j++) {
            if (wp->objects[j].cost < min_cost) {
                min_cost = wp->objects[j].cost;
                min_cost_idx = j;
            }
        }
        if (min_cost <= wp->cost) {
            TFWO_Object *best_obj = &wp->objects[min_cost_idx];
            double *temp_pos = data->temp_RR; // Reuse RR
            double *wp_pos = wp->position;
            double *obj_pos = best_obj->position;
            double sum = 0.0;
            for (int k = 0; k < dim; k++) {
                temp_pos[k] = wp_pos[k];
                wp_pos[k] = obj_pos[k];
                obj_pos[k] = temp_pos[k];
                sum += wp_pos[k];
            }
            wp->position_sum = sum;
            double temp_cost = wp->cost;
            wp->cost = min_cost;
            best_obj->cost = temp_cost;
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
    free(data->temp_d);
    free(data->temp_d2);
    free(data->temp_RR);
    free(data->temp_J);
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
    data->temp_d = (double *)calloc(opt->dim, sizeof(double));
    data->temp_d2 = (double *)calloc(opt->dim, sizeof(double));
    data->temp_RR = (double *)calloc(opt->dim, sizeof(double));
    data->temp_J = (double *)calloc(data->n_whirlpools, sizeof(double));
    if (!data->best_costs || !data->mean_costs || !data->temp_d || !data->temp_d2 || !data->temp_RR || !data->temp_J) {
        fprintf(stderr, "Error: Failed to allocate arrays\n");
        free(data->best_costs);
        free(data->mean_costs);
        free(data->temp_d);
        free(data->temp_d2);
        free(data->temp_RR);
        free(data->temp_J);
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
            double cost = data->whirlpools[i].cost;
            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
            mean_cost += cost;
        }
        mean_cost /= data->n_whirlpools;

        // Update global best
        if (best_cost < opt->best_solution.fitness) {
            opt->best_solution.fitness = best_cost;
            double *dst = opt->best_solution.position;
            double *src = data->whirlpools[best_idx].position;
            for (int k = 0; k < opt->dim; k++) {
                dst[k] = src[k];
            }
        }

        // Store iteration results
        data->best_costs[iter] = best_cost;
        data->mean_costs[iter] = mean_cost;

        printf("Iter %d: Best Cost = %f\n", iter + 1, best_cost);
    }

    free_tfwo_data(data);
}
