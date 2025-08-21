#define JA_CRUISING_PROBABILITY 0.8f
#define JA_CRUISING_DISTANCE 0.1f
#define JA_ALPHA 0.1f
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}
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
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }
    fitness[gid] = INFINITY;
}
__kernel void cruising_phase(
    __global float *population,
    __global float *bounds,
    __global uint *seeds,
    float cruising_distance,
    int dim,
    int num_cruising
) {
    int gid = get_global_id(0);
    if (gid >= num_cruising) return;
    __global float *pos = population + gid * dim;
    float direction[32];
    float norm = 0.0f;
    for (int j = 0; j < dim; j++) {
        direction[j] = rand_float(&seeds[gid]) * 2.0f - 1.0f;
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);
    if (norm == 0.0f) norm = 1.0f;
    for (int j = 0; j < dim; j++) {
        direction[j] /= norm;
        pos[j] += JA_ALPHA * cruising_distance * direction[j];
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}
__kernel void random_walk_phase(
    __global float *population,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int num_cruising,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid < num_cruising || gid >= pop_size) return;
    __global float *pos = population + gid * dim;
    float direction[32];
    float norm = 0.0f;
    for (int j = 0; j < dim; j++) {
        direction[j] = rand_float(&seeds[gid]) * 2.0f - 1.0f;
        norm += direction[j] * direction[j];
    }
    norm = sqrt(norm);
    if (norm == 0.0f) norm = 1.0f;
    for (int j = 0; j < dim; j++) {
        direction[j] /= norm;
        pos[j] += JA_ALPHA * direction[j];
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}
