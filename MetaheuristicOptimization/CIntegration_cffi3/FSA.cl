#define RAND_MAX 4294967295.0f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / RAND_MAX;
}

// Initialize population
__kernel void initialize_population(
    __global float *population,
    __global float *fitness,
    __global float *local_best_positions,
    __global float *local_best_values,
    __global float *bounds,
    __global float *best_position,
    __global float *best_fitness,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Initialize position
    __global float *pos = population + gid * dim;
    __global float *local_best = local_best_positions + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        local_best[j] = pos[j];
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
    local_best_values[gid] = INFINITY;
}

// Update population
__kernel void update_population(
    __global float *population,
    __global float *fitness,
    __global float *local_best_positions,
    __global float *local_best_values,
    __global float *best_position,
    __global float *best_fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *local_best = local_best_positions + gid * dim;
    __global float *temp_pos = pos; // Use pos as temp buffer for new position
    float current_fitness = fitness[gid];

    // Update rule: Move towards global and local bests
    for (int j = 0; j < dim; j++) {
        float delta_global = (best_position[j] - pos[j]) * rand_float(&seeds[gid]);
        float delta_local = (local_best[j] - pos[j]) * rand_float(&seeds[gid]);
        temp_pos[j] = pos[j] + delta_global + delta_local;

        // Enforce bounds
        temp_pos[j] = clamp(temp_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Initial strategy update
__kernel void initial_strategy_update(
    __global float *population,
    __global float *fitness,
    __global float *local_best_positions,
    __global float *local_best_values,
    __global float *best_position,
    __global float *best_fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *local_best = local_best_positions + gid * dim;
    __global float *temp_pos = pos; // Use pos as temp buffer for new position

    // Update rule: Move towards global best with influence from local best
    for (int j = 0; j < dim; j++) {
        temp_pos[j] = best_position[j] + (best_position[j] - local_best[j]) * rand_float(&seeds[gid]);

        // Enforce bounds
        temp_pos[j] = clamp(temp_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
