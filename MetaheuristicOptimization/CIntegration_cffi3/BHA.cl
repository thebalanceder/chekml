#define EPSILON 1e-10f

// Xorshift random number generator
uint xorshift32(__global uint *seeds, int idx) {
    uint x = seeds[idx];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    seeds[idx] = x;
    return x;
}

float rand_float(__global uint *seeds, int idx) {
    return (float)xorshift32(seeds, idx) / (float)UINT_MAX;
}

// Kernel to initialize stars
__kernel void init_stars(__global float *positions,
                         __global float *bounds,
                         __global uint *seeds,
                         int dim,
                         int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    for (int j = 0; j < dim; j++) {
        float min = bounds[2 * j];
        float max = bounds[2 * j + 1];
        positions[idx * dim + j] = min + (max - min) * rand_float(seeds, idx);
    }
}

// Kernel to update star positions
__kernel void update_positions(__global float *positions,
                              __global float *bounds,
                              __global uint *seeds,
                              int dim,
                              int population_size,
                              int black_hole_idx) {
    int idx = get_global_id(0);
    if (idx >= population_size || idx == black_hole_idx) return;

    int base_idx = idx * dim;
    int bh_base_idx = black_hole_idx * dim;
    float rand_val = rand_float(seeds, idx);

    for (int j = 0; j < dim; j++) {
        positions[base_idx + j] += rand_val * (positions[bh_base_idx + j] - positions[base_idx + j]);
        // Enforce bounds
        if (positions[base_idx + j] < bounds[2 * j]) {
            positions[base_idx + j] = bounds[2 * j];
        } else if (positions[base_idx + j] > bounds[2 * j + 1]) {
            positions[base_idx + j] = bounds[2 * j + 1];
        }
    }
}

// Kernel to generate new stars
__kernel void new_star_gen(__global float *positions,
                          __global float *fitness,
                          __global float *bounds,
                          __global uint *seeds,
                          int dim,
                          int population_size,
                          int black_hole_idx,
                          float best_fitness) {
    int idx = get_global_id(0);
    if (idx >= population_size || idx == black_hole_idx) return;

    // Compute total fitness
    float total_fitness = 0.0f;
    for (int i = 0; i < population_size; i++) {
        total_fitness += fitness[i];
    }
    float R = best_fitness / (total_fitness + EPSILON);
    float R_squared = R * R;

    // Check if star crosses event horizon
    int base_idx = idx * dim;
    int bh_base_idx = black_hole_idx * dim;
    float dist_squared = 0.0f;
    for (int j = 0; j < dim; j++) {
        float diff = positions[bh_base_idx + j] - positions[base_idx + j];
        dist_squared += diff * diff;
    }

    if (dist_squared < R_squared) {
        for (int j = 0; j < dim; j++) {
            float min = bounds[2 * j];
            float max = bounds[2 * j + 1];
            positions[base_idx + j] = min + (max - min) * rand_float(seeds, idx);
        }
    }
}

// Kernel to find black hole (min fitness)
__kernel void find_black_hole(__global float *fitness,
                              __global int *best_index,
                              int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    __local float local_min_fitness[256]; // Adjust size based on work-group
    __local int local_min_idx[256];

    local_min_fitness[idx] = fitness[idx];
    local_min_idx[idx] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (idx < stride && idx + stride < get_local_size(0)) {
            if (local_min_fitness[idx] > local_min_fitness[idx + stride]) {
                local_min_fitness[idx] = local_min_fitness[idx + stride];
                local_min_idx[idx] = local_min_idx[idx + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx == 0) {
        *best_index = local_min_idx[0];
    }
}
