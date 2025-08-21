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
    __global float *alpha_position,
    __global float *beta_position,
    __global float *delta_position,
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

// Update wolf positions
__kernel void update_positions(
    __global float *population,
    __global float *fitness,
    __global float *alpha_position,
    __global float *beta_position,
    __global float *delta_position,
    __global float *bounds,
    __global uint *seeds,
    float a,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    
    for (int j = 0; j < dim; j++) {
        // Alpha update
        float r1 = rand_float(&seeds[gid]);
        float r2 = rand_float(&seeds[gid]);
        float A1 = 2.0f * a * r1 - a;
        float C1 = 2.0f * r2;
        float D_alpha = fabs(C1 * alpha_position[j] - pos[j]);
        float X1 = alpha_position[j] - A1 * D_alpha;

        // Beta update
        r1 = rand_float(&seeds[gid]);
        r2 = rand_float(&seeds[gid]);
        float A2 = 2.0f * a * r1 - a;
        float C2 = 2.0f * r2;
        float D_beta = fabs(C2 * beta_position[j] - pos[j]);
        float X2 = beta_position[j] - A2 * D_beta;

        // Delta update
        r1 = rand_float(&seeds[gid]);
        r2 = rand_float(&seeds[gid]);
        float A3 = 2.0f * a * r1 - a;
        float C3 = 2.0f * r2;
        float D_delta = fabs(C3 * delta_position[j] - pos[j]);
        float X3 = delta_position[j] - A3 * D_delta;

        // Update position
        pos[j] = (X1 + X2 + X3) / 3.0f;

        // Enforce bounds
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
