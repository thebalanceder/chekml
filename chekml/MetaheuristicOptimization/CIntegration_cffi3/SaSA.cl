#define SASA_C1_FACTOR 2.0f
#define SASA_C1_EXPONENT 4.0f

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

// Update salp positions
__kernel void update_salps(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    float c1,
    __global float *best_position,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;

    if (gid < pop_size / 2) {
        // Leader salp update (Eq. 3.1)
        for (int j = 0; j < dim; j++) {
            float lb = bounds[j * 2];
            float ub = bounds[j * 2 + 1];
            float c2 = rand_float(&seeds[gid]);
            float c3 = rand_float(&seeds[gid]);
            float new_position;
            if (c3 < 0.5f) {
                new_position = best_position[j] + c1 * ((ub - lb) * c2 + lb);
            } else {
                new_position = best_position[j] - c1 * ((ub - lb) * c2 + lb);
            }
            pos[j] = clamp(new_position, lb, ub);
        }
    } else {
        // Follower salp update (Eq. 3.4)
        __global float *prev_pos = population + (gid - 1) * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] = (prev_pos[j] + pos[j]) / 2.0f;
            float lb = bounds[j * 2];
            float ub = bounds[j * 2 + 1];
            pos[j] = clamp(pos[j], lb, ub);
        }
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
