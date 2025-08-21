#define BBO_ALPHA 0.9f
#define MUTATION_PROB 0.1f
#define MUTATION_SCALE 0.02f

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
    __global float *mu,
    __global float *lambda_,
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

    // Initialize migration rates
    mu[gid] = 1.0f - ((float)gid / (pop_size - 1));
    lambda_[gid] = 1.0f - mu[gid];

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Migration phase
__kernel void migration_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *mu,
    __global float *lambda_,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float lambda = lambda_[gid];

    for (int j = 0; j < dim; j++) {
        if (rand_float(&seeds[gid]) <= lambda) {
            // Compute emigration probabilities
            float ep_sum = 0.0f;
            float ep[32]; // Assuming pop_size <= 32 for simplicity
            for (int k = 0; k < pop_size; k++) {
                ep[k] = (k == gid) ? 0.0f : mu[k];
                ep_sum += ep[k];
            }
            if (ep_sum > 0.0f) {
                // Normalize probabilities
                for (int k = 0; k < pop_size; k++) {
                    ep[k] /= ep_sum;
                }
                // Roulette wheel selection
                float r = rand_float(&seeds[gid]);
                float cumsum = 0.0f;
                int source_idx = pop_size - 1;
                for (int k = 0; k < pop_size; k++) {
                    cumsum += ep[k];
                    if (r <= cumsum) {
                        source_idx = k;
                        break;
                    }
                }
                // Migration step
                __global float *source_pos = population + source_idx * dim;
                pos[j] += BBO_ALPHA * (source_pos[j] - pos[j]);
            }
        }
        // Enforce bounds
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Mutation phase
__kernel void mutation_phase(
    __global float *population,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float scale[32]; // Assuming dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        scale[j] = (bounds[j * 2 + 1] - bounds[j * 2]) * MUTATION_SCALE;
    }

    for (int j = 0; j < dim; j++) {
        if (rand_float(&seeds[gid]) <= MUTATION_PROB) {
            // Approximate normal distribution using Box-Muller transform
            float u1 = rand_float(&seeds[gid]);
            float u2 = rand_float(&seeds[gid]);
            float z = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265359f * u2);
            pos[j] += scale[j] * z;
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }
}
