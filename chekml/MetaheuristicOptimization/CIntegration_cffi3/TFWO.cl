#define PI 3.14159265358979323846f

// Xorshift random number generator
uint xorshift32(__global uint *seeds, int idx) {
    uint x = seeds[idx];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    seeds[idx] = x;
    return x;
}

float rand_float(__global uint *seeds, int idx) {
    return (float)xorshift32(seeds, idx) / (float)UINT_MAX;
}

// Kernel to initialize whirlpools and objects
__kernel void init_whirlpools(__global float *wp_positions,
                             __global float *wp_costs,
                             __global float *wp_deltas,
                             __global float *wp_position_sums,
                             __global float *obj_positions,
                             __global float *obj_costs,
                             __global float *obj_deltas,
                             __global float *bounds,
                             __global uint *seeds,
                             int dim,
                             int n_whirlpools,
                             int n_objects_per_whirlpool) {
    int idx = get_global_id(0);
    int total_objects = n_whirlpools * n_objects_per_whirlpool;

    if (idx < n_whirlpools) {
        // Initialize whirlpool
        int base_idx = idx * dim;
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            float min = bounds[2 * j];
            float max = bounds[2 * j + 1];
            float pos = min + (max - min) * rand_float(seeds, idx);
            wp_positions[base_idx + j] = pos;
            sum += pos;
        }
        wp_position_sums[idx] = sum;
        wp_deltas[idx] = 0.0f;
        wp_costs[idx] = 0.0f; // To be set by CPU
    } else if (idx < n_whirlpools + total_objects) {
        // Initialize object
        int obj_idx = idx - n_whirlpools;
        int base_idx = obj_idx * dim;
        for (int j = 0; j < dim; j++) {
            float min = bounds[2 * j];
            float max = bounds[2 * j + 1];
            float pos = min + (max - min) * rand_float(seeds, idx);
            obj_positions[base_idx + j] = pos;
        }
        obj_deltas[obj_idx] = 0.0f;
        obj_costs[obj_idx] = 0.0f; // To be set by CPU
    }
}

// Kernel to compute whirlpool effects
__kernel void effects_whirlpools(__global float *wp_positions,
                                __global float *wp_costs,
                                __global float *wp_deltas,
                                __global float *wp_position_sums,
                                __global float *obj_positions,
                                __global float *obj_costs,
                                __global float *obj_deltas,
                                __global float *bounds,
                                __global uint *seeds,
                                __global float *temp_d,
                                __global float *temp_d2,
                                __global float *temp_RR,
                                __global float *temp_J,
                                int dim,
                                int n_whirlpools,
                                int n_objects_per_whirlpool,
                                int iteration) {
    int idx = get_global_id(0);
    int total_objects = n_whirlpools * n_objects_per_whirlpool;

    if (idx >= n_whirlpools && idx < n_whirlpools + total_objects) {
        // Update object
        int obj_idx = idx - n_whirlpools;
        int wp_idx = obj_idx / n_objects_per_whirlpool;
        int obj_idx_in_wp = obj_idx % n_objects_per_whirlpool;
        int obj_base_idx = obj_idx * dim;
        int wp_base_idx = wp_idx * dim;

        // Compute sum of object position
        float sum_obj = 0.0f;
        for (int k = 0; k < dim; k++) {
            sum_obj += obj_positions[obj_base_idx + k];
        }

        // Find influencing whirlpools
        int min_idx = wp_idx, max_idx = wp_idx;
        if (n_whirlpools > 1) {
            for (int t = 0; t < n_whirlpools; t++) {
                if (t != wp_idx) {
                    temp_J[t] = fabs(wp_costs[t]) * sqrt(fabs(wp_position_sums[t] - sum_obj));
                } else {
                    temp_J[t] = INFINITY;
                }
            }
            float min_J = INFINITY, max_J = -INFINITY;
            for (int t = 0; t < n_whirlpools; t++) {
                if (temp_J[t] < min_J && t != wp_idx) {
                    min_J = temp_J[t];
                    min_idx = t;
                }
                if (temp_J[t] > max_J && t != wp_idx) {
                    max_J = temp_J[t];
                    max_idx = t;
                }
            }
        }

        // Compute d and d2
        int min_base_idx = min_idx * dim;
        int max_base_idx = max_idx * dim;
        for (int k = 0; k < dim; k++) {
            temp_d[obj_base_idx + k] = rand_float(seeds, idx) * (wp_positions[min_base_idx + k] - obj_positions[obj_base_idx + k]);
            temp_d2[obj_base_idx + k] = (n_whirlpools > 1) ? rand_float(seeds, idx) * (wp_positions[max_base_idx + k] - obj_positions[obj_base_idx + k]) : 0.0f;
        }

        // Update delta
        obj_deltas[obj_idx] += rand_float(seeds, idx) * rand_float(seeds, idx) * PI;
        float eee = obj_deltas[obj_idx];
        float cos_eee = cos(eee);
        float sin_eee = sin(eee);
        float fr0 = cos_eee;
        float fr10 = -sin_eee;
        float fr0_fr10 = fabs(fr0 * fr10);

        // Compute new position
        for (int k = 0; k < dim; k++) {
            float x = (fr0 * temp_d[obj_base_idx + k] + fr10 * temp_d2[obj_base_idx + k]) * (1.0f + fr0_fr10);
            temp_RR[obj_base_idx + k] = wp_positions[wp_base_idx + k] - x;
            // Enforce bounds
            temp_RR[obj_base_idx + k] = max(bounds[2 * k], min(bounds[2 * k + 1], temp_RR[obj_base_idx + k]));
        }

        // Random jump
        float cos_eee_sq = cos_eee * cos_eee;
        float sin_eee_sq = sin_eee * sin_eee;
        float FE_i = (cos_eee_sq * sin_eee_sq) * (cos_eee_sq * sin_eee_sq);
        if (rand_float(seeds, idx) < FE_i) {
            int k = xorshift32(seeds, idx) % dim;
            obj_positions[obj_base_idx + k] = bounds[2 * k] + (bounds[2 * k + 1] - bounds[2 * k]) * rand_float(seeds, idx);
        } else {
            // Update position if better (cost comparison done on CPU)
            for (int k = 0; k < dim; k++) {
                obj_positions[obj_base_idx + k] = temp_RR[obj_base_idx + k];
            }
        }
    } else if (idx < n_whirlpools) {
        // Update whirlpool
        int wp_base_idx = idx * dim;
        float sum_i = wp_position_sums[idx];

        // Find nearest whirlpool
        int min_idx = 0;
        float min_J = INFINITY;
        for (int t = 0; t < n_whirlpools; t++) {
            if (t != idx) {
                float J = wp_costs[t] * fabs(wp_position_sums[t] - sum_i);
                if (J < min_J) {
                    min_J = J;
                    min_idx = t;
                }
            }
        }

        // Update delta
        wp_deltas[idx] += rand_float(seeds, idx) * rand_float(seeds, idx) * PI;
        float fr = fabs(cos(wp_deltas[idx]) + sin(wp_deltas[idx]));

        // Compute new position
        int min_base_idx = min_idx * dim;
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            float x = fr * rand_float(seeds, idx) * (wp_positions[min_base_idx + k] - wp_positions[wp_base_idx + k]);
            temp_RR[wp_base_idx + k] = wp_positions[min_base_idx + k] - x;
            sum += temp_RR[wp_base_idx + k];
            // Enforce bounds
            temp_RR[wp_base_idx + k] = max(bounds[2 * k], min(bounds[2 * k + 1], temp_RR[wp_base_idx + k]));
        }
        wp_position_sums[idx] = sum;
        for (int k = 0; k < dim; k++) {
            wp_positions[wp_base_idx + k] = temp_RR[wp_base_idx + k];
        }
    }
}

// Kernel to update best whirlpool
__kernel void update_best(__global float *wp_positions,
                         __global float *wp_costs,
                         __global float *wp_position_sums,
                         __global float *obj_positions,
                         __global float *obj_costs,
                         __global float *temp_RR,
                         int dim,
                         int n_whirlpools,
                         int n_objects_per_whirlpool) {
    int idx = get_global_id(0);
    if (idx >= n_whirlpools) return;

    int wp_base_idx = idx * dim;
    int obj_base = idx * n_objects_per_whirlpool;
    float min_cost = obj_costs[obj_base];
    int min_cost_idx = obj_base;

    // Find best object in whirlpool
    for (int j = 1; j < n_objects_per_whirlpool; j++) {
        int obj_idx = obj_base + j;
        if (obj_costs[obj_idx] < min_cost) {
            min_cost = obj_costs[obj_idx];
            min_cost_idx = obj_idx;
        }
    }

    // Update whirlpool if object is better
    if (min_cost <= wp_costs[idx]) {
        int obj_base_idx = min_cost_idx * dim;
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            temp_RR[wp_base_idx + k] = wp_positions[wp_base_idx + k];
            wp_positions[wp_base_idx + k] = obj_positions[obj_base_idx + k];
            obj_positions[obj_base_idx + k] = temp_RR[wp_base_idx + k];
            sum += wp_positions[wp_base_idx + k];
        }
        wp_position_sums[idx] = sum;
        float temp_cost = wp_costs[idx];
        wp_costs[idx] = min_cost;
        obj_costs[min_cost_idx] = temp_cost;
    }
}
