#define ICA_ASSIMILATION_COEFF 2.0f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize countries
__kernel void initialize_countries(
    __global float *positions,
    __global float *costs,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Initialize position
    __global float *pos = positions + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder cost (to be evaluated on CPU)
    costs[gid] = INFINITY;
}

// Assimilate colonies
__kernel void assimilate_colonies(
    __global float *positions,
    __global int *empire_indices,
    __global float *bounds,
    __global uint *seeds,
    float assim_coeff,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Check if country is a colony (non-imperialist)
    int empire_idx = empire_indices[gid];
    if (empire_idx == -1) return; // Imperialist, skip

    // Find imperialist position
    int imp_idx = -1;
    for (int i = 0; i < pop_size; i++) {
        if (empire_indices[i] == empire_idx && i != gid) {
            imp_idx = i;
            break;
        }
    }
    if (imp_idx == -1) return;

    __global float *pos = positions + gid * dim;
    __global float *imp_pos = positions + imp_idx * dim;
    for (int j = 0; j < dim; j++) {
        float dist = imp_pos[j] - pos[j];
        pos[j] += assim_coeff * rand_float(&seeds[gid]) * dist;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}
