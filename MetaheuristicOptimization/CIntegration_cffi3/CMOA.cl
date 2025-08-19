#define CMOA_CROSSOVER_RATE 0.5f
#define CMOA_MUTATION_RATE 0.3f
#define CMOA_MUTATION_SCALE 0.1f

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

inline int find_closest_solution(
    __global float *population,
    int gid,
    int dim,
    int pop_size
) {
    float min_distance = INFINITY;
    int closest_idx = gid;
    __global float *pos_i = population + gid * dim;
    for (int k = 0; k < pop_size; k++) {
        if (k == gid) continue;
        float distance = 0.0f;
        __global float *pos_k = population + k * dim;
        for (int j = 0; j < dim; j++) {
            distance += fabs(pos_k[j] - pos_i[j]);
        }
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = k;
        }
    }
    return closest_idx;
}

__kernel void update_solutions(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    float cos_t,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;
    __global float *pos = population + gid * dim;
    float randoms[7];
    for (int r = 0; r < 7; r++) {
        randoms[r] = (r < 5 ? rand_float(&seeds[gid]) : (rand_float(&seeds[gid]) * 2.0f - 1.0f));
    }
    int idx1 = (int)(rand_float(&seeds[gid]) * pop_size);
    int idx2;
    do {
        idx2 = (int)(rand_float(&seeds[gid]) * pop_size);
    } while (idx2 == idx1);
    if (randoms[0] < CMOA_CROSSOVER_RATE) {
        int closest_idx = find_closest_solution(population, gid, dim, pop_size);
        __global float *closest_pos = population + closest_idx * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] += randoms[1] * (closest_pos[j] - pos[j]);
        }
    }
    for (int j = 0; j < dim; j++) {
        pos[j] += randoms[2] * (best_position[j] - pos[j]);
    }
    for (int j = 0; j < dim; j++) {
        pos[j] += cos_t * randoms[3] * (best_position[j] - pos[j]);
    }
    if (randoms[4] < CMOA_MUTATION_RATE) {
        for (int j = 0; j < dim; j++) {
            float range = bounds[j * 2 + 1] - bounds[j * 2];
            pos[j] += randoms[5] * CMOA_MUTATION_SCALE * range;
        }
    }
    __global float *pos1 = population + idx1 * dim;
    __global float *pos2 = population + idx2 * dim;
    for (int j = 0; j < dim; j++) {
        pos[j] += randoms[6] * (pos1[j] - pos2[j]);
    }
    for (int j = 0; j < dim; j++) {
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    fitness[gid] = INFINITY;
}
