#define SEO_RAND_SCALE 0.1f

// Random number generator (XOR-shift)
inline float rand_float(__global uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Approximate normal random number using Central Limit Theorem
inline float rand_normal(__global uint *seed) {
    float sum = 0.0f;
    for (int i = 0; i < 12; i++) {
        sum += rand_float(seed);
    }
    return sum - 6.0f; // Mean 0, approximate std 1
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

// Social engineering update
__kernel void social_engineering_update(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    float scale,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *current = population + gid * dim;

    // Select a random target agent
    int target_index = (int)(rand_float(&seeds[gid]) * pop_size);
    while (target_index == gid) {
        target_index = (int)(rand_float(&seeds[gid]) * pop_size);
    }

    __global float *target = population + target_index * dim;

    // Update position with social engineering formula
    for (int j = 0; j < dim; j++) {
        float diff = target[j] - current[j];
        float randn = rand_normal(&seeds[gid]) * scale;
        current[j] += randn * diff;
        current[j] = clamp(current[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
