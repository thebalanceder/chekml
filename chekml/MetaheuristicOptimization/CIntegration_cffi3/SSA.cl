#define SSA_MAX_GLIDING_DISTANCE 1.11f
#define SSA_MIN_GLIDING_DISTANCE 0.5f
#define SSA_GLIDING_CONSTANT 1.9f
#define SSA_HICKORY_NUT_TREE 1
#define SSA_ACORN_NUT_TREE 3

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize squirrel population
__kernel void initialize_population(
    __global float *population,
    __global float *velocities,
    __global float *fitness,
    __global int *tree_types,
    __global float *pulse_flying_rates,
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
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Initialize velocity
    __global float *vel = velocities + gid * dim;
    for (int j = 0; j < dim; j++) {
        vel[j] = 0.0f;
    }

    // Initialize tree type (1: hickory, 2: normal, 3: acorn)
    float r = rand_float(&seeds[gid]);
    if (r < 0.02f) {
        tree_types[gid] = SSA_HICKORY_NUT_TREE; // ~1/50 for hickory
    } else if (r < 0.06f) {
        tree_types[gid] = SSA_ACORN_NUT_TREE;   // ~2/50 for acorn
    } else {
        tree_types[gid] = 2;                    // ~47/50 for normal
    }

    // Initialize pulse flying rate
    pulse_flying_rates[gid] = rand_float(&seeds[gid]);

    // Placeholder fitness
    fitness[gid] = INFINITY;
}

// Update squirrel positions
__kernel void update_squirrels(
    __global float *population,
    __global float *velocities,
    __global float *fitness,
    __global int *tree_types,
    __global float *pulse_flying_rates,
    __global float *best_position,
    __global float *bounds,
    __global uint *seeds,
    float mean_A,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *vel = velocities + gid * dim;
    float gliding_diff = SSA_MAX_GLIDING_DISTANCE - SSA_MIN_GLIDING_DISTANCE;
    float gliding_distance = SSA_MIN_GLIDING_DISTANCE + gliding_diff * rand_float(&seeds[gid]);
    int tree_type = tree_types[gid];
    float multiplier = (tree_type == SSA_HICKORY_NUT_TREE) ? 3.0f : (tree_type == SSA_ACORN_NUT_TREE) ? 1.0f : 2.0f;

    // Update velocity and position
    for (int j = 0; j < dim; j++) {
        vel[j] += gliding_distance * SSA_GLIDING_CONSTANT * (best_position[j] - pos[j]) * multiplier;
        pos[j] += vel[j];
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Random flying condition
    if (rand_float(&seeds[gid]) > pulse_flying_rates[gid]) {
        float eps = -1.0f + 2.0f * rand_float(&seeds[gid]);
        for (int j = 0; j < dim; j++) {
            pos[j] = best_position[j] + eps * mean_A;
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Placeholder fitness
    fitness[gid] = INFINITY;
}
