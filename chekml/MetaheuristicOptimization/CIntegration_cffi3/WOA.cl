// File: WOA.cl
#define WOA_A_INITIAL 2.0f
#define WOA_A2_INITIAL -1.0f
#define WOA_A2_FINAL -2.0f
#define WOA_B 1.0f

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

// Update whale positions
__kernel void update_whales(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    float a,
    float a2,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float r1 = rand_float(&seeds[gid]);
    float r2 = rand_float(&seeds[gid]);
    float A = 2.0f * a * r1 - a;  // Eq. (2.3)
    float C = 2.0f * r2;          // Eq. (2.4)
    float l = (a2 - 1.0f) * rand_float(&seeds[gid]) + 1.0f;  // Parameter in Eq. (2.5)
    float p = rand_float(&seeds[gid]);  // Strategy selection

    // Select a random whale for exploration
    int rand_idx = (int)(rand_float(&seeds[gid]) * pop_size);
    if (rand_idx == gid) rand_idx = (rand_idx + 1) % pop_size;  // Avoid selecting self

    for (int j = 0; j < dim; j++) {
        if (p < 0.5f) {
            if (fabs(A) >= 1.0f) {  // Search for prey (exploration)
                __global float *rand_pos = population + rand_idx * dim;
                float D_X_rand = fabs(C * rand_pos[j] - pos[j]);  // Eq. (2.7)
                pos[j] = rand_pos[j] - A * D_X_rand;  // Eq. (2.8)
            } else {  // Encircling prey (exploitation)
                float D_Leader = fabs(C * best_position[j] - pos[j]);  // Eq. (2.1)
                pos[j] = best_position[j] - A * D_Leader;  // Eq. (2.2)
            }
        } else {  // Spiral bubble-net attack
            float distance2Leader = fabs(best_position[j] - pos[j]);
            pos[j] = distance2Leader * exp(WOA_B * l) * cos(l * 2.0f * M_PI) + best_position[j];  // Eq. (2.5)
        }
        // Enforce bounds
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
