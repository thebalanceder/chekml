#define POA_STEP_SIZE 0.1f

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
    __global float *best_position,
    __global float *best_fitness,
    __global uint *random_seeds,
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
        pos[j] = lower + rand_float(&random_seeds[gid]) * (upper - lower);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Update particle positions
__kernel void update_positions(
    __global float *population,
    __global float *fitness,
    __global float *best_position,
    __global float *bounds,
    __global uint *random_seeds,
    float step_size,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;

    // Calculate direction and norm
    float norm = 0.0f;
    float direction[32]; // Assume dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        direction[j] = best_position[j] - pos[j];
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);

    // Update position with step size and bounds clamping
    if (norm > 0.0f) {
        float inv_norm = step_size / norm;
        for (int j = 0; j < dim; j++) {
            pos[j] += direction[j] * inv_norm;
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
