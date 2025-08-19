#define EFO_RANDOMIZATION_RATE 0.3f
#define EFO_POSITIVE_SELECTION_RATE 0.2f
#define EFO_GOLDEN_RATIO 1.618033988749895f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize population
__kernel void initialize_population(
    __global float *position,
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
    __global float *pos = position + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Generate new particle
__kernel void generate_new_particle(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global float *new_particle,
    float randomization_rate,
    float positive_selection_rate,
    float golden_ratio,
    int positive_field_size,
    int negative_field_start,
    int neutral_field_start,
    int neutral_field_end,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid != 0) return; // Only one particle is generated

    float r = rand_float(&seeds[gid]);
    float rp = rand_float(&seeds[gid]);
    float randomization = rand_float(&seeds[gid]);

    // Precompute random indices
    int r_index1 = (int)(rand_float(&seeds[gid]) * positive_field_size);
    int r_index2 = negative_field_start + (int)(rand_float(&seeds[gid]) * (pop_size - negative_field_start));
    int r_index3 = neutral_field_start + (int)(rand_float(&seeds[gid]) * (neutral_field_end - neutral_field_start));

    for (int i = 0; i < dim; i++) {
        float ps = rand_float(&seeds[gid]);
        if (ps > positive_selection_rate) {
            // Use particles from positive, neutral, and negative fields
            new_particle[i] = (population[r_index3 * dim + i] +
                              golden_ratio * r * (population[r_index1 * dim + i] - population[r_index3 * dim + i]) +
                              r * (population[r_index3 * dim + i] - population[r_index2 * dim + i]));
        } else {
            // Copy from positive field
            new_particle[i] = population[r_index1 * dim + i];
        }

        // Check boundaries
        if (new_particle[i] < bounds[i * 2] || new_particle[i] > bounds[i * 2 + 1]) {
            new_particle[i] = bounds[i * 2] + (bounds[i * 2 + 1] - bounds[i * 2]) * randomization;
        }
    }

    // Randomize one dimension with probability RANDOMIZATION_RATE
    if (rp < randomization_rate) {
        int ri = (int)(rand_float(&seeds[gid]) * dim);
        new_particle[ri] = bounds[ri * 2] + (bounds[ri * 2 + 1] - bounds[ri * 2]) * randomization;
    }
}
