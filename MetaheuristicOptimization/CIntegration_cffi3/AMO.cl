#define AMO_NEIGHBORHOOD_SIZE 5
#define AMO_MIGRATION_PROBABILITY_FACTOR 0.5f

// XOR-shift random number generator
inline float rand_float(__global uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Normal random number using Box-Muller transform
inline float normal_rand(__global uint *seed) {
    float u = rand_float(seed);
    float v = rand_float(seed);
    if (u == 0.0f) u = 1e-6f; // Avoid log(0)
    float s = sqrt(-2.0f * log(u));
    return s * cos(2.0f * M_PI_F * v);
}

// Initialize population
__kernel void initialize_population(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }
    fitness[gid] = INFINITY; // Placeholder fitness
}

// Neighborhood learning phase
__kernel void neighborhood_learning_phase(
    __global float *population,
    __global uint *seeds,
    __global float *bounds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float new_pos[32]; // Assuming dim <= 32 for local storage
    int lseq[AMO_NEIGHBORHOOD_SIZE];

    // Define neighborhood indices
    if (gid == 0) {
        lseq[0] = pop_size - 2;
        lseq[1] = pop_size - 1;
        lseq[2] = gid;
        lseq[3] = gid + 1;
        lseq[4] = gid + 2;
    } else if (gid == 1) {
        lseq[0] = pop_size - 1;
        lseq[1] = gid - 1;
        lseq[2] = gid;
        lseq[3] = gid + 1;
        lseq[4] = gid + 2;
    } else if (gid == pop_size - 2) {
        lseq[0] = gid - 2;
        lseq[1] = gid - 1;
        lseq[2] = gid;
        lseq[3] = pop_size - 1;
        lseq[4] = 0;
    } else if (gid == pop_size - 1) {
        lseq[0] = gid - 2;
        lseq[1] = gid - 1;
        lseq[2] = gid;
        lseq[3] = 0;
        lseq[4] = 1;
    } else {
        lseq[0] = gid - 2;
        lseq[1] = gid - 1;
        lseq[2] = gid;
        lseq[3] = gid + 1;
        lseq[4] = gid + 2;
    }

    // Random permutation of neighborhood
    for (int j = 0; j < AMO_NEIGHBORHOOD_SIZE; j++) {
        int idx = (int)(rand_float(&seeds[gid]) * AMO_NEIGHBORHOOD_SIZE);
        int temp = lseq[j];
        lseq[j] = lseq[idx];
        lseq[idx] = temp;
    }

    int exemplar_idx = lseq[1];
    __global float *exemplar_pos = population + exemplar_idx * dim;
    float FF = normal_rand(&seeds[gid]);

    // Compute new position
    for (int j = 0; j < dim; j++) {
        new_pos[j] = pos[j] + FF * (exemplar_pos[j] - pos[j]);
        new_pos[j] = clamp(new_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Update population
    for (int j = 0; j < dim; j++) {
        pos[j] = new_pos[j];
    }
}

// Global migration phase
__kernel void global_migration_phase(
    __global float *population,
    __global float *fitness,
    __global float *best_position,
    __global uint *seeds,
    __global float *bounds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float new_pos[32]; // Assuming dim <= 32

    // Compute probability based on fitness rank (approximated)
    float probability = AMO_MIGRATION_PROBABILITY_FACTOR; // Simplified for GPU

    // Select random indices r1 and r3
    int indices[32]; // Assuming pop_size <= 32 for simplicity
    for (int j = 0; j < pop_size; j++) indices[j] = j;

    // Fisher-Yates shuffle
    for (int j = pop_size - 1; j > 0; j--) {
        int k = (int)(rand_float(&seeds[gid]) * (j + 1));
        int temp = indices[j];
        indices[j] = indices[k];
        indices[k] = temp;
    }

    int idx = 0;
    while (indices[idx] == gid) idx++;
    int r1 = indices[idx++];
    while (indices[idx] == gid) idx++;
    int r3 = indices[idx];

    __global float *pos_r1 = population + r1 * dim;
    __global float *pos_r3 = population + r3 * dim;

    // Update position
    if (rand_float(&seeds[gid]) > probability) {
        for (int j = 0; j < dim; j++) {
            new_pos[j] = pos_r1[j] +
                         rand_float(&seeds[gid]) * (best_position[j] - pos[j]) +
                         rand_float(&seeds[gid]) * (pos_r3[j] - pos[j]);
            new_pos[j] = clamp(new_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        for (int j = 0; j < dim; j++) {
            new_pos[j] = pos[j];
        }
    }

    // Update population
    for (int j = 0; j < dim; j++) {
        pos[j] = new_pos[j];
    }
}
