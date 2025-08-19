#define WGMO_ALPHA 0.9f
#define WGMO_BETA 0.1f
#define WGMO_GAMMA 0.1f

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

// Update geese positions
__kernel void update_geese(
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

    for (int j = 0; j < dim; j++) {
        float r_beta = rand_float(&seeds[gid]);
        float r_gamma = rand_float(&seeds[gid]);
        float bounds_diff = bounds[j * 2 + 1] - bounds[j * 2];

        // Update position
        pos[j] = WGMO_ALPHA * pos[j] +
                 WGMO_BETA * r_beta * (best_position[j] - pos[j]) +
                 WGMO_GAMMA * r_gamma * bounds_diff;

        // Enforce bounds
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
