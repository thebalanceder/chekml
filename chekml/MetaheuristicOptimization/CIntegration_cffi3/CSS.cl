#define KA 1.0f
#define KV 1.0f
#define A 1.0f
#define EPSILON 1e-10f

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

// Kernel to initialize particles
__kernel void init_particles(__global float *positions,
                            __global float *bounds,
                            __global uint *seeds,
                            int dim,
                            int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    for (int j = 0; j < dim; j++) {
        float min = bounds[2 * j];
        float max = bounds[2 * j + 1];
        positions[idx * dim + j] = min + (max - min) * rand_float(seeds, idx);
    }
}

// Kernel to calculate forces
__kernel void calc_forces(__global float *positions,
                         __global float *costs,
                         __global float *forces,
                         __global float *bounds,
                         __global uint *seeds,
                         int dim,
                         int population_size,
                         float best_cost) {
    int i = get_global_id(0);
    if (i >= population_size) return;

    // Initialize forces to zero
    int i_base = i * dim;
    for (int k = 0; k < dim; k++) {
        forces[i_base + k] = 0.0f;
    }

    // Find min/max costs
    float fitbest = best_cost, fitworst = -INFINITY;
    for (int k = 0; k < population_size; k++) {
        if (costs[k] > fitworst) fitworst = costs[k];
    }

    // Compute charge for particle i
    float charge_i;
    if (fabs(fitbest - fitworst) < EPSILON) {
        charge_i = 1.0f;
    } else {
        charge_i = (costs[i] - fitworst) / (fitbest - fitworst);
    }

    // Calculate forces
    for (int j = 0; j < population_size; j++) {
        if (j == i) continue; // Skip self-interaction
        float r_ij = 0.0f, r_ij_norm = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = positions[i_base + k] - positions[j * dim + k];
            r_ij += diff * diff;
            float norm_diff = (positions[i_base + k] + positions[j * dim + k]) / 2.0f - positions[0]; // Approximate best position
            r_ij_norm += norm_diff * norm_diff;
        }
        r_ij = sqrt(r_ij);
        r_ij_norm = r_ij / (sqrt(r_ij_norm) + EPSILON);

        float charge_j = (fabs(fitbest - fitworst) < EPSILON) ? 1.0f : (costs[j] - fitworst) / (fitbest - fitworst);
        float p_ij = (costs[i] < costs[j] || 
                     ((costs[i] - fitbest) / (fitworst - fitbest + EPSILON) > rand_float(seeds, i))) ? 1.0f : 0.0f;
        float force_term = (r_ij < A) ? (charge_i * charge_j * r_ij / (A * A * A)) : (charge_i * charge_j / (r_ij * r_ij));

        for (int k = 0; k < dim; k++) {
            float force = p_ij * force_term * (positions[i_base + k] - positions[j * dim + k]);
            forces[i_base + k] += force;
        }
    }
}

// Kernel to update positions
__kernel void update_positions(__global float *positions,
                              __global float *forces,
                              __global float *velocities,
                              __global float *bounds,
                              __global uint *seeds,
                              int dim,
                              int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    int base_idx = idx * dim;
    float rand1 = rand_float(seeds, idx);
    float rand2 = rand_float(seeds, idx);
    float dt = 1.0f;

    for (int j = 0; j < dim; j++) {
        int idx_j = base_idx + j;
        velocities[idx_j] = rand1 * KV * velocities[idx_j] + rand2 * KA * forces[idx_j];
        positions[idx_j] += velocities[idx_j] * dt;

        // Enforce bounds
        if (positions[idx_j] < bounds[2 * j]) {
            positions[idx_j] = bounds[2 * j];
            velocities[idx_j] = 0.0f; // Reset velocity at boundary
        } else if (positions[idx_j] > bounds[2 * j + 1]) {
            positions[idx_j] = bounds[2 * j + 1];
            velocities[idx_j] = 0.0f; // Reset velocity at boundary
        }
    }
}

// Kernel to update charged memory
__kernel void update_cm(__global float *positions,
                       __global float *costs,
                       __global float *cm_positions,
                       __global float *cm_costs,
                       int dim,
                       int population_size,
                       int cm_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    // Use local memory to find top cm_size costs
    __local float local_costs[256]; // Adjust size based on work-group
    __local int local_indices[256];
    local_costs[idx] = costs[idx];
    local_indices[idx] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction to find top cm_size
    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (idx < stride && idx + stride < get_local_size(0)) {
            if (local_costs[idx] > local_costs[idx + stride]) {
                float temp_cost = local_costs[idx];
                int temp_idx = local_indices[idx];
                local_costs[idx] = local_costs[idx + stride];
                local_indices[idx] = local_indices[idx + stride];
                local_costs[idx + stride] = temp_cost;
                local_indices[idx + stride] = temp_idx;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Copy top solutions to charged memory
    if (idx < cm_size && local_indices[idx] < population_size) {
        int src_idx = local_indices[idx];
        for (int j = 0; j < dim; j++) {
            cm_positions[idx * dim + j] = positions[src_idx * dim + j];
        }
        cm_costs[idx] = costs[src_idx];
    }
}

// Kernel to find min/max indices
__kernel void find_min_max(__global float *costs,
                          __global int *min_max_indices,
                          int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    __local float local_min_cost;
    __local float local_max_cost;
    __local int local_min_idx;
    __local int local_max_idx;

    if (idx == 0) {
        local_min_cost = costs[0];
        local_max_cost = costs[0];
        local_min_idx = 0;
        local_max_idx = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (costs[idx] < local_min_cost) {
        local_min_cost = costs[idx];
        local_min_idx = idx;
    }
    if (costs[idx] > local_max_cost) {
        local_max_cost = costs[idx];
        local_max_idx = idx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx == 0) {
        min_max_indices[0] = local_min_idx;
        min_max_indices[1] = local_max_idx;
    }
}
