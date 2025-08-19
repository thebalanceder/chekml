#define HPO_CONSTRICTION_COEFF 0.1f
#define HPO_C_PARAM_MAX 0.98f
#define HPO_TWO_PI 6.283185307179586f

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

// Update positions (Safe and Attack modes)
__kernel void update_positions(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global float *xi,
    __global float *dist,
    __global int *idxsortdist,
    __global uint *seeds,
    int iter,
    float c_factor,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float c = 1.0f - ((float)iter * c_factor);
    int kbest = (int)(pop_size * c + 0.5f);

    // Compute mean position (xi) on CPU, so just read it
    // Compute distance to mean for this individual
    float dist_val = 0.0f;
    for (int j = 0; j < dim; j++) {
        float diff = xi[j] - pos[j];
        dist_val += diff * diff;
    }
    dist_val = sqrt(dist_val);
    dist[gid] = dist_val;
    idxsortdist[gid] = gid;

    // Sorting done on CPU, so assume idxsortdist is valid
    float r2 = rand_float(&seeds[gid]);
    float z[32]; // Assuming dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        float r1 = rand_float(&seeds[gid]) < c ? 1.0f : 0.0f;
        float r3 = rand_float(&seeds[gid]);
        z[j] = r1 == 0.0f ? r2 : r3;
    }

    if (rand_float(&seeds[gid]) < HPO_CONSTRICTION_COEFF) {
        // Safe mode: Move towards mean and kbest-th closest individual
        int si_idx = idxsortdist[kbest - 1];
        __global float *si_pos = population + si_idx * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] += 0.5f * (
                (2.0f * c * z[j] * si_pos[j] - pos[j]) +
                (2.0f * (1.0f - c) * z[j] * xi[j] - pos[j])
            );
        }
    } else {
        // Attack mode: Move towards target with cosine perturbation
        for (int j = 0; j < dim; j++) {
            float rr = -1.0f + 2.0f * z[j];
            pos[j] = 2.0f * z[j] * cos(HPO_TWO_PI * rr) * 
                     (best_position[j] - pos[j]) + best_position[j];
        }
    }

    // Enforce bounds
    for (int j = 0; j < dim; j++) {
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
