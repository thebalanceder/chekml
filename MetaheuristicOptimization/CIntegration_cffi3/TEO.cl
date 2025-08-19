#define TEO_STEP_SIZE 0.1f

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

// Perturb and accept solutions
__kernel void perturb_and_accept(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    float temperature,
    __global float *best_position,
    float best_fitness,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;

    // Compute scale for perturbation
    float scale[32]; // Assuming dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        scale[j] = fabs(bounds[j * 2 + 1] - bounds[j * 2]);
    }

    // Perturb the current solution
    for (int j = 0; j < dim; j++) {
        float perturbation = TEO_STEP_SIZE * (rand_float(&seeds[gid]) - 0.5f + rand_float(&seeds[gid]) - 0.5f) * scale[j];
        pos[j] = best_position[j] + perturbation;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
