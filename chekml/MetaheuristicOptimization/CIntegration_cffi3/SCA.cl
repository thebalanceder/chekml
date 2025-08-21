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
__kernel void update_positions(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    float r1_factor,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;
    __global float *pos = population + gid * dim;
    const float two_pi = 6.283185307179586f;
    for (int j = 0; j < dim; j++) {
        float r2 = two_pi * rand_float(&seeds[gid]);
        float r3 = 2.0f * rand_float(&seeds[gid]);
        float r4 = rand_float(&seeds[gid]);
        float delta = r3 * best_position[j] - pos[j];
        pos[j] += r1_factor * (r4 < 0.5f ? sin(r2) : cos(r2)) * fabs(delta);
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    fitness[gid] = INFINITY;
}
