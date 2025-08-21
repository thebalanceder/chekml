inline float rand_float(__global uint *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (float)(*seed) / (float)0x7fffffffu;
}

__kernel void initialize_population(__global float *positions, __global float *bounds,
                                   __global uint *seeds, int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        positions[gid * dim + j] = lower + (upper - lower) * rand_float(&seeds[gid]);
    }
}

__kernel void compute_mean(__global float *positions, __global float *mean_student,
                          int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= dim) return;

    float sum = 0.0f;
    for (int i = 0; i < population_size; i++) {
        sum += positions[i * dim + gid];
    }
    mean_student[gid] = sum / population_size;
}

__kernel void teacher_phase(__global float *positions, __global float *costs,
                           __global float *bounds, __global uint *seeds,
                           __global float *mean_student, int best_idx,
                           int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    int tf = 1 + (rand_float(&seeds[gid]) < 0.5f); // Teaching factor (1 or 2)
    for (int j = 0; j < dim; j++) {
        float current = positions[gid * dim + j];
        float teacher = positions[best_idx * dim + j];
        float mean = mean_student[j];
        float new_pos = current + rand_float(&seeds[gid]) * (teacher - tf * mean);
        new_pos = max(bounds[j * 2], min(bounds[j * 2 + 1], new_pos));
        positions[gid * dim + j] = new_pos;
    }
}

__kernel void learner_phase(__global float *positions, __global float *costs,
                           __global float *bounds, __global uint *seeds,
                           __global int *partners, int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    int partner_idx = partners[gid];
    float current_cost = costs[gid];
    float partner_cost = costs[partner_idx];

    for (int j = 0; j < dim; j++) {
        float current = positions[gid * dim + j];
        float partner = positions[partner_idx * dim + j];
        float new_pos;
        if (current_cost < partner_cost) {
            new_pos = current + rand_float(&seeds[gid]) * (current - partner);
        } else {
            new_pos = current + rand_float(&seeds[gid]) * (partner - current);
        }
        new_pos = max(bounds[j * 2], min(bounds[j * 2 + 1], new_pos));
        positions[gid * dim + j] = new_pos;
    }
}
