#define OSA_STEP_SIZE 0.1f
#define OSA_P_EXPLORE 0.1f

// XOR-shift random number generator
inline float rand_float(__global uint *seed) {
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
    __global float *best_position,
    __global float *best_fitness,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        pos[j] = lb + rand_float(&seeds[gid]) * (ub - lb);
    }
    fitness[gid] = INFINITY;
}

// Update population
__kernel void update_population(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;

    if (rand_float(&seeds[gid]) < OSA_P_EXPLORE) {
        // Exploration: Random movement
        for (int j = 0; j < dim; j++) {
            float move = OSA_STEP_SIZE * (2.0f * rand_float(&seeds[gid]) - 1.0f);
            float new_pos = pos[j] + move;
            float lb = bounds[j * 2];
            float ub = bounds[j * 2 + 1];
            pos[j] = fmax(lb, fmin(ub, new_pos));
        }
    } else {
        // Exploitation: Move towards best
        for (int j = 0; j < dim; j++) {
            float direction = best_position[j] - pos[j];
            float new_pos = pos[j] + OSA_STEP_SIZE * direction;
            float lb = bounds[j * 2];
            float ub = bounds[j * 2 + 1];
            pos[j] = fmax(lb, fmin(ub, new_pos));
        }
    }
    fitness[gid] = INFINITY;
}
