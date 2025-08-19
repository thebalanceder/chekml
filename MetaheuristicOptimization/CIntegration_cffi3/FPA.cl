// Constants
#define FPA_SWITCH_PROB 0.8f
#define FPA_LEVY_STEP_SCALE 0.01f
#define FPA_LEVY_BETA 1.5f
#define FPA_LEVY_SIGMA 0.6966f
#define FPA_INV_LEVY_BETA (1.0f / 1.5f)

// Random number generator
float rand_float(uint *seed) {
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
    return (float)(*seed) / (float)0x7fffffff;
}

// Normal distribution approximation
float rand_normal(uint *seed) {
    float sum = 0.0f;
    for (int i = 0; i < 12; i++) {
        sum += rand_float(seed);
    }
    return sum - 6.0f; // Mean 0, variance 1
}

// Lévy flight step
float levy_step(uint *seed) {
    float u = rand_normal(seed) * FPA_LEVY_SIGMA;
    float v = rand_normal(seed);
    return FPA_LEVY_STEP_SCALE * u / pow(fabs(v), FPA_INV_LEVY_BETA);
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

    uint seed = seeds[gid];
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        population[gid * dim + j] = lower + rand_float(&seed) * (upper - lower);
    }
    seeds[gid] = seed;
    fitness[gid] = INFINITY;

    if (gid == 0) {
        *best_fitness = INFINITY;
        for (int j = 0; j < dim; j++) {
            best_position[j] = population[j];
        }
    }
}

__kernel void global_pollination_phase(
    __global float *population,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid];
    if (rand_float(&seed) > FPA_SWITCH_PROB) {
        for (int j = 0; j < dim; j++) {
            float step = levy_step(&seed);
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            population[gid * dim + j] += step * (population[gid * dim + j] - best_position[j]);
            population[gid * dim + j] = clamp(population[gid * dim + j], lower, upper);
        }
    }
    seeds[gid] = seed;
}

__kernel void local_pollination_phase(
    __global float *population,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid];
    if (rand_float(&seed) <= FPA_SWITCH_PROB) {
        // Select two random indices
        int j_idx = (int)(rand_float(&seed) * pop_size);
        int k_idx = (int)(rand_float(&seed) * pop_size);
        while (j_idx == k_idx) {
            k_idx = (int)(rand_float(&seed) * pop_size);
        }

        float epsilon = rand_float(&seed);
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            population[gid * dim + j] += epsilon * (population[j_idx * dim + j] - population[k_idx * dim + j]);
            population[gid * dim + j] = clamp(population[gid * dim + j], lower, upper);
        }
    }
    seeds[gid] = seed;
}
