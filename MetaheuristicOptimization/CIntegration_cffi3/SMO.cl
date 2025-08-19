#define SMO_PERTURBATION_RATE 0.1f
#define SMO_BHC_DELTA 0.1f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize population
__kernel void initialize_population(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Initialize position
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Local Leader Phase
__kernel void local_leader_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *group_ids,
    __global float *group_leaders,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    int group_id = group_ids[gid];
    if (rand_float(&seeds[gid]) > SMO_PERTURBATION_RATE) {
        // Select random member from same group
        int rand_idx = gid;
        int attempts = 0;
        while ((rand_idx == gid || group_ids[rand_idx] != group_id) && attempts < 10) {
            rand_idx = (int)(rand_float(&seeds[gid]) * pop_size);
            attempts++;
        }
        if (group_ids[rand_idx] != group_id) return; // Skip if no valid member found

        __global float *pos = population + gid * dim;
        __global float *rand_pos = population + rand_idx * dim;
        __global float *leader_pos = group_leaders + group_id * dim;

        for (int j = 0; j < dim; j++) {
            float new_pos = pos[j] +
                            (leader_pos[j] - pos[j]) * rand_float(&seeds[gid]) +
                            (pos[j] - rand_pos[j]) * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            pos[j] = clamp(new_pos, bounds[j * 2], bounds[j * 2 + 1]);
        }
        fitness[gid] = INFINITY; // Mark for CPU evaluation
    }
}

// Global Leader Phase
__kernel void global_leader_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *group_ids,
    __global float *global_leader,
    float max_fitness,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    int group_id = group_ids[gid];
    float prob = fitness[gid] / (max_fitness + 1e-10f); // Avoid division by zero
    if (rand_float(&seeds[gid]) < prob) {
        // Select random member from same group
        int rand_idx = gid;
        int attempts = 0;
        while ((rand_idx == gid || group_ids[rand_idx] != group_id) && attempts < 10) {
            rand_idx = (int)(rand_float(&seeds[gid]) * pop_size);
            attempts++;
        }
        if (group_ids[rand_idx] != group_id) return; // Skip if no valid member found

        __global float *pos = population + gid * dim;
        __global float *rand_pos = population + rand_idx * dim;

        for (int j = 0; j < dim; j++) {
            float new_pos = pos[j] +
                            (global_leader[j] - pos[j]) * rand_float(&seeds[gid]) +
                            (pos[j] - rand_pos[j]) * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            pos[j] = clamp(new_pos, bounds[j * 2], bounds[j * 2 + 1]);
        }
        fitness[gid] = INFINITY; // Mark for CPU evaluation
    }
}

// Beta-Hill Climbing
__kernel void beta_hill_climbing(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float delta = SMO_BHC_DELTA * (bounds[j * 2 + 1] - bounds[j * 2]);
        float new_pos = pos[j] + (rand_float(&seeds[gid]) * 2.0f - 1.0f) * delta;
        pos[j] = clamp(new_pos, bounds[j * 2], bounds[j * 2 + 1]);
    }
    fitness[gid] = INFINITY; // Mark for CPU evaluation
}
