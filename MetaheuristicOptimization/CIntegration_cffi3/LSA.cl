#define LSA_ENERGY_FACTOR 2.05f
#define FOCKING_PROB 0.002f
#define DIRECTION_STEP 0.0f
#define MAX_DIM 100

// Random number generator (XOR-shift)
inline float random_float(__global uint *seed) {
    uint s = *seed;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *seed = s;
    return s / 4294967296.0f;
}

// Normal distribution (simplified CLT approximation)
float rand_normal(__global uint *seed, float mean, float stddev) {
    float sum = 0.0f;
    for (int i = 0; i < 12; i++) {
        sum += random_float(seed);
    }
    return mean + stddev * (sum - 6.0f);
}

// Exponential distribution
float rand_exponential(__global uint *seed, float lambda) {
    float u = random_float(seed);
    return -log(1.0f - u) / lambda;
}

// Initialize channels
__kernel void initialize_channels(
    __global float *position,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size)
{
    int i = get_global_id(0);
    if (i >= pop_size) return;

    __global float *pos = position + i * dim;
    for (int j = 0; j < dim; j++) {
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        pos[j] = lb + (ub - lb) * random_float(&seeds[i]);
    }
    fitness[i] = INFINITY;
}

// Update positions
__kernel void lsa_update_positions(
    __global float *position,
    __global float *fitness,
    __global float *directions,
    __global float *bounds,
    __global uint *seeds,
    int best_idx,
    float energy,
    int dim,
    int pop_size)
{
    int i = get_global_id(0);
    if (i >= pop_size || dim > MAX_DIM) return;

    __global float *pos = position + i * dim;
    __global float *best_pos = position + best_idx * dim;
    float temp_pos[MAX_DIM];

    float is_best = 1.0f;
    for (int j = 0; j < dim; j++) {
        if (fabs(pos[j] - best_pos[j]) > 1e-6f) {
            is_best = 0.0f;
            break;
        }
    }

    for (int j = 0; j < dim; j++) {
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        if (is_best == 1.0f) {
            temp_pos[j] = pos[j] + directions[j] * fabs(rand_normal(&seeds[i], 0.0f, energy));
        } else {
            float dist = pos[j] - best_pos[j];
            float r = rand_exponential(&seeds[i], fabs(dist));
            temp_pos[j] = pos[j] + (dist < 0.0f ? r : -r);
        }
        // Clamp to bounds
        temp_pos[j] = clamp(temp_pos[j], lb, ub);
    }

    // Update position
    for (int j = 0; j < dim; j++) {
        pos[j] = temp_pos[j];
    }

    // Focking procedure (simplified)
    if (random_float(&seeds[i]) < FOCKING_PROB) {
        for (int j = 0; j < dim; j++) {
            float lb = bounds[j * 2];
            float ub = bounds[j * 2 + 1];
            pos[j] = lb + ub - pos[j];
        }
    }
}
