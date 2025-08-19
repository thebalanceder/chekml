#define A_MAX 1.0f
#define A_MIN 0.0f

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
    __global float *leaders,
    __global float *leader_fitness,
    __global float *pbest,
    __global float *pbest_fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size,
    int num_leaders
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Initialize population position
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        pos[j] = lb + (ub - lb) * rand_float(&seeds[gid]);
    }

    // Initialize fitness
    fitness[gid] = INFINITY;

    // Initialize pbest
    __global float *pb_pos = pbest + gid * dim;
    for (int j = 0; j < dim; j++) {
        pb_pos[j] = pos[j];
    }
    pbest_fitness[gid] = INFINITY;

    // Initialize leaders (first num_leaders agents)
    if (gid < num_leaders) {
        __global float *leader_pos = leaders + gid * dim;
        for (int j = 0; j < dim; j++) {
            leader_pos[j] = pos[j];
        }
        leader_fitness[gid] = INFINITY;
    }
}

// Update positions using GWO
__kernel void update_positions_gwo(
    __global float *population,
    __global float *leaders,
    __global float *bounds,
    __global uint *seeds,
    float a,
    int dim,
    int pop_size,
    int num_leaders
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;

    for (int j = 0; j < dim; j++) {
        float sum_XX = 0.0f;
        for (int k = 0; k < num_leaders; k++) {
            float r1 = rand_float(&seeds[gid]);
            float r2 = rand_float(&seeds[gid]);
            float A1 = 2.0f * a * r1 - a;
            float C1 = 2.0f * r2;
            float leader_pos = leaders[k * dim + j];
            float D_alpha = fabs(C1 * leader_pos - pos[j]);
            sum_XX += leader_pos - A1 * D_alpha;
        }
        pos[j] = sum_XX / (float)num_leaders;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}
