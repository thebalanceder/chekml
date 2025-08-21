#define MSO_MAX_P_EXPLORE 0.8f
#define MSO_MIN_P_EXPLORE 0.1f
#define MSO_PERTURBATION_SCALE 10.0f
#define M_PI_F 3.14159265359f
inline float rand_float(__global uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}
inline float rand_normal(float mean, float stddev, __global uint *seed) {
    float u1 = rand_float(seed);
    float u2 = rand_float(seed);
    float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    return mean + stddev * z;
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
    __global float *best_position,
    __global float *bounds,
    __global uint *seeds,
    int iter,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;
    __global float *pos = population + gid * dim;
    float dynamic_p_explore = fmax(MSO_MAX_P_EXPLORE * exp(-0.1f * iter), MSO_MIN_P_EXPLORE);
    for (int j = 0; j < dim; j++) {
        float r = rand_float(&seeds[gid]);
        float rand_val = rand_float(&seeds[gid]);
        if (rand_val < dynamic_p_explore) {
            if (r < 0.5f) {
                pos[j] = best_position[j] + rand_float(&seeds[gid]) * (bounds[j * 2 + 1] - best_position[j]);
            } else {
                pos[j] = best_position[j] - rand_float(&seeds[gid]) * (best_position[j] - bounds[j * 2]);
            }
        } else {
            float perturbation = rand_normal(0.0f, 1.0f, &seeds[gid]) * (bounds[j * 2 + 1] - bounds[j * 2]) / MSO_PERTURBATION_SCALE;
            pos[j] = best_position[j] + perturbation;
        }
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    fitness[gid] = INFINITY;
}
