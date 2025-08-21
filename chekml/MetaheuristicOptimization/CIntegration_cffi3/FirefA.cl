#define FA_BETA0 1.0f
#define FA_GAMMA 0.01f

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

// Update firefly positions
__kernel void update_fireflies(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    float alpha,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float current_fitness = fitness[gid];

    // Compute scale for randomization
    float scale[32]; // Assuming dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        scale[j] = fabs(bounds[j * 2 + 1] - bounds[j * 2]);
    }

    // Compare with all other fireflies
    for (int j = 0; j < pop_size; j++) {
        if (j == gid) continue;
        if (current_fitness >= fitness[j]) { // Move if j is brighter
            __global float *other_pos = population + j * dim;
            float r = 0.0f;
            for (int k = 0; k < dim; k++) {
                r += (pos[k] - other_pos[k]) * (pos[k] - other_pos[k]);
            }
            r = sqrt(r);
            float beta = FA_BETA0 * exp(-FA_GAMMA * r * r);
            for (int k = 0; k < dim; k++) {
                float step = alpha * (rand_float(&seeds[gid]) - 0.5f) * scale[k];
                pos[k] += beta * (other_pos[k] - pos[k]) + step;
                pos[k] = clamp(pos[k], bounds[k * 2], bounds[k * 2 + 1]);
            }
        }
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
