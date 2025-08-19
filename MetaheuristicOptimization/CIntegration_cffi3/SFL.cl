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

// Update memeplex (simplified for one offspring per subcomplex)
__kernel void update_memeplex(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    __global int *memeplex_indices,
    __global float *probabilities,
    int dim,
    int memeplex_size,
    int num_parents,
    int memeplex_id
) {
    int gid = get_global_id(0);
    if (gid >= memeplex_size) return;

    int seed_idx = memeplex_id * memeplex_size + gid;
    float r = rand_float(&seeds[seed_idx]);

    // Select parent using precomputed probabilities
    float cumsum = 0.0f;
    int parent_idx = 0;
    for (int i = 0; i < memeplex_size; i++) {
        cumsum += probabilities[i];
        if (r <= cumsum) {
            parent_idx = i;
            break;
        }
    }

    // Only process if selected as part of subcomplex
    if (parent_idx < num_parents) {
        __global float *pos = population + memeplex_indices[memeplex_id * memeplex_size + parent_idx] * dim;
        float current_fitness = fitness[memeplex_indices[memeplex_id * memeplex_size + parent_idx]];

        // Simplified: Move worst towards best in memeplex
        if (gid == num_parents - 1) { // Worst in subcomplex
            __global float *best_pos = population + memeplex_indices[memeplex_id * memeplex_size] * dim; // Best in memeplex
            for (int j = 0; j < dim; j++) {
                float step = 2.0f * rand_float(&seeds[seed_idx]) * (best_pos[j] - pos[j]);
                pos[j] += step; // Fixed: Removed erroneous "pos" before pos[j]
                pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
            }
            fitness[memeplex_indices[memeplex_id * memeplex_size + parent_idx]] = INFINITY; // Mark for CPU evaluation
        }
    }
}
