#define RDA_STEP_SIZE 0.1f
#define RDA_P_EXPLORATION 0.1f

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

// Update positions
__kernel void update_positions(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    float step_size,
    float p_exploration,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float r = rand_float(&seeds[gid]);

    if (r < p_exploration) {
        // Exploration: Move randomly
        for (int j = 0; j < dim; j++) {
            pos[j] += step_size * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        // Exploitation: Move towards the best solution
        for (int j = 0; j < dim; j++) {
            pos[j] += step_size * (best_position[j] - pos[j]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
