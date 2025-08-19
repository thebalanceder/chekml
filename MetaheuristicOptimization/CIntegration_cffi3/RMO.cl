#define RMO_ALPHA 0.1f

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

    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }
    fitness[gid] = INFINITY;
}

// Parallel prefix sum for reference point calculation
__kernel void compute_reference_point(
    __global float *population,
    __global float *reference_point,
    int dim,
    int pop_size,
    __local float *local_sums
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    if (gid >= pop_size) return;

    for (int j = 0; j < dim; j++) {
        local_sums[lid * dim + j] = population[gid * dim + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (lid < offset && (lid + offset) < group_size) {
            for (int j = 0; j < dim; j++) {
                local_sums[lid * dim + j] += local_sums[(lid + offset) * dim + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        for (int j = 0; j < dim; j++) {
            reference_point[j] = local_sums[j] / pop_size;
        }
    }
}

// Update positions
__kernel void update_reference_point_and_positions(
    __global float *population,
    __global float *bounds,
    __global float *reference_point,
    float alpha,
    int pop_size,
    int dim
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float direction = reference_point[j] - pos[j];
        pos[j] += alpha * direction;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}

// Parallel bitonic sort for population based on fitness
__kernel void sort_population(
    __global float *population,
    __global float *fitness,
    int pop_size,
    int dim
) {
    int gid = get_global_id(0);
    if (gid >= pop_size / 2) return;

    // Bitonic sort step (simplified for parallel execution)
    for (int stage = 0; stage < pop_size; stage++) {
        for (int substage = 0; substage < stage; substage++) {
            int partner = gid ^ (1 << substage);
            if (partner >= pop_size) continue;

            bool ascending = ((gid & (1 << stage)) == 0);
            float fitness1 = fitness[gid];
            float fitness2 = fitness[partner];

            if ((ascending && fitness1 > fitness2) || (!ascending && fitness1 < fitness2)) {
                // Swap fitness values
                float temp_fitness = fitness[gid];
                fitness[gid] = fitness[partner];
                fitness[partner] = temp_fitness;

                // Swap positions
                for (int j = 0; j < dim; j++) {
                    float temp_pos = population[gid * dim + j];
                    population[gid * dim + j] = population[partner * dim + j];
                    population[partner * dim + j] = temp_pos;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
