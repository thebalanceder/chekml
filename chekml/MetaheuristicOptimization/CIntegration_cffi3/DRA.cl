#define DRA_BELIEF_PROFILE_RATE 0.5f
#define DRA_MIRACLE_RATE 0.5f
#define DRA_PROSELYTISM_RATE 0.9f
#define DRA_REWARD_PENALTY_RATE 0.2f
#define DRA_NUM_GROUPS 5
#define PI 3.141592653589793f

inline float rand_float(__global uint *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (float)(*seed) / (float)0x7fffffffu;
}

__kernel void initialize_belief_profiles(__global float *positions, __global float *bounds,
                                        __global uint *seeds, int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        positions[gid * dim + j] = lower + (upper - lower) * rand_float(&seeds[gid]);
    }
}

__kernel void find_min_max(__global float *costs, __global int *min_max_indices, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    // Simplified: each work item updates shared min/max indices (not fully parallel, needs reduction)
    float cost = costs[gid];
    if (cost < costs[min_max_indices[0]]) {
        min_max_indices[0] = gid;
    }
    if (cost > costs[min_max_indices[1]]) {
        min_max_indices[1] = gid;
    }
}

__kernel void miracle_operator(__global float *positions, __global float *costs,
                              __global float *bounds, __global uint *seeds,
                              int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    float rand_val = rand_float(&seeds[gid]);
    float new_position[32]; // Assume max dim <= 32 for simplicity

    for (int j = 0; j < dim; j++) {
        new_position[j] = positions[gid * dim + j];
    }

    if (rand_val <= 0.5f) {
        float factor = cos(PI / 2.0f) * (rand_float(&seeds[gid]) - cos(rand_float(&seeds[gid])));
        for (int j = 0; j < dim; j++) {
            new_position[j] *= factor;
        }
    } else {
        float r = rand_float(&seeds[gid]);
        for (int j = 0; j < dim; j++) {
            new_position[j] += r * (new_position[j] - round(pow(1.0f, r)) * new_position[j]);
        }
    }

    // Enforce bounds
    for (int j = 0; j < dim; j++) {
        new_position[j] = max(bounds[j * 2], min(bounds[j * 2 + 1], new_position[j]));
    }

    // Update if better (fitness evaluated on CPU)
    for (int j = 0; j < dim; j++) {
        positions[gid * dim + j] = new_position[j];
    }
}

__kernel void proselytism_operator(__global float *positions, __global float *costs,
                                  __global float *bounds, __global uint *seeds,
                                  int min_idx, int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    float rand_val = rand_float(&seeds[gid]);
    float new_position[32]; // Assume max dim <= 32

    for (int j = 0; j < dim; j++) {
        new_position[j] = positions[gid * dim + j];
    }

    if (rand_val > (1.0f - DRA_MIRACLE_RATE)) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += new_position[j];
        }
        float mean_bp = sum / dim;
        for (int j = 0; j < dim; j++) {
            new_position[j] = (new_position[j] * 0.01f +
                               mean_bp * (1.0f - DRA_MIRACLE_RATE) +
                               (1.0f - mean_bp) -
                               (rand_float(&seeds[gid]) - 4.0f * sin(sin(PI * rand_float(&seeds[gid])))));
        }
    } else {
        for (int j = 0; j < dim; j++) {
            new_position[j] = positions[min_idx * dim + j] * (rand_float(&seeds[gid]) - cos(rand_float(&seeds[gid])));
        }
    }

    // Enforce bounds
    for (int j = 0; j < dim; j++) {
        new_position[j] = max(bounds[j * 2], min(bounds[j * 2 + 1], new_position[j]));
    }

    // Update if better (fitness evaluated on CPU)
    for (int j = 0; j < dim; j++) {
        positions[gid * dim + j] = new_position[j];
    }
}

__kernel void reward_penalty_operator(__global float *positions, __global float *costs,
                                     __global float *bounds, __global uint *seeds,
                                     int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    // Randomly select one individual per work item
    float rand_val = rand_float(&seeds[gid]);
    if (rand_val > 0.5f) return; // Reduce number of updates for efficiency

    float new_position[32]; // Assume max dim <= 32
    for (int j = 0; j < dim; j++) {
        new_position[j] = positions[gid * dim + j];
    }

    float factor = (rand_val >= DRA_REWARD_PENALTY_RATE) ? (1.0f - rand_float(&seeds[gid])) : (1.0f + rand_float(&seeds[gid]));
    for (int j = 0; j < dim; j++) {
        new_position[j] *= factor;
    }

    // Enforce bounds
    for (int j = 0; j < dim; j++) {
        new_position[j] = max(bounds[j * 2], min(bounds[j * 2 + 1], new_position[j]));
    }

    // Update if better (fitness evaluated on CPU)
    for (int j = 0; j < dim; j++) {
        positions[gid * dim + j] = new_position[j];
    }
}

__kernel void replacement_operator(__global float *positions, __global float *costs,
                                  __global uint *seeds, int dim, int population_size, int num_groups) {
    int gid = get_global_id(0);
    if (gid >= num_groups) return;

    // Swap missionary (index gid) with a random follower
    int follower_idx = (int)(rand_float(&seeds[gid]) * (population_size - num_groups)) + num_groups;
    if (follower_idx >= population_size) return;

    // Swap positions
    for (int j = 0; j < dim; j++) {
        float temp = positions[gid * dim + j];
        positions[gid * dim + j] = positions[follower_idx * dim + j];
        positions[follower_idx * dim + j] = temp;
    }

    // Fitness updated on CPU
}
