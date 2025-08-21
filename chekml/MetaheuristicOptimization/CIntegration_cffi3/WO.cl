#define WO_FEMALE_PROPORTION 0.4f
#define WO_HALTON_BASE 7
#define WO_LEVY_BETA 1.5f

// Random number generator (XOR-shift)
inline float rand_float(__private uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Halton sequence
inline float halton_sequence(int index, int base) {
    float result = 0.0f;
    float f = 1.0f / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i = i / base;
        f /= base;
    }
    return result;
}

// Levy flight step
inline void levy_flight(__global float *step, int dim, float sigma, __private uint *seed) {
    uint private_seed = *seed; // Copy seed to private memory
    for (int i = 0; i < dim; i++) {
        float r = rand_float(&private_seed);
        float s = rand_float(&private_seed);
        float u = sigma * sqrt(-2.0f * log(r)) * cos(2.0f * M_PI * s);
        float v = sqrt(-2.0f * log(r)) * sin(2.0f * M_PI * s);
        step[i] = u / pow(fabs(v), 1.0f / WO_LEVY_BETA);
    }
    *seed = private_seed; // Update private seed
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

    // Initialize position
    __global float *pos = population + gid * dim;
    uint seed = seeds[gid]; // Copy seed
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seed) * (upper - lower);
    }
    seeds[gid] = seed; // Update seed
    fitness[gid] = INFINITY;
}

// Migration phase
__kernel void migration_phase(
    __global float *population,
    __global int *temp_indices,
    __global float *bounds,
    __global uint *seeds,
    float beta,
    float r3,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid]; // Copy seed
    int j = (int)(rand_float(&seed) * pop_size);
    if (j == pop_size) j--;

    float step = beta * r3 * r3;
    __global float *pos = population + gid * dim;
    __global float *other_pos = population + temp_indices[j] * dim;
    for (int k = 0; k < dim; k++) {
        pos[k] += step * (other_pos[k] - pos[k]);
        pos[k] = clamp(pos[k], bounds[k * 2], bounds[k * 2 + 1]);
    }
    seeds[gid] = seed; // Update seed
}

// Male position update
__kernel void male_position_update(
    __global float *population,
    __global float *bounds,
    int male_count,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= male_count) return;

    float halton_val = halton_sequence(gid + 1, WO_HALTON_BASE);
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + halton_val * (upper - lower);
        pos[j] = clamp(pos[j], lower, upper);
    }
}

// Female position update
__kernel void female_position_update(
    __global float *population,
    __global float *best_position,
    __global float *bounds,
    float alpha,
    int male_count,
    int female_count,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= female_count) return;

    int idx = male_count + gid;
    float one_minus_alpha = 1.0f - alpha;
    __global float *pos = population + idx * dim;
    __global float *male_pos = population + (idx - male_count) * dim;
    for (int j = 0; j < dim; j++) {
        pos[j] += alpha * (male_pos[j] - pos[j]) + one_minus_alpha * (best_position[j] - pos[j]);
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}

// Child position update
__kernel void child_position_update(
    __global float *population,
    __global float *best_position,
    __global float *bounds,
    __global uint *seeds,
    __global float *temp_array1,
    __global float *temp_array2,
    float levy_sigma,
    int child_count,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= child_count) return;

    int idx = pop_size - child_count + gid;
    uint seed = seeds[gid]; // Copy seed
    float P = rand_float(&seed);
    levy_flight(temp_array1, dim, levy_sigma, &seed);
    __global float *pos = population + idx * dim;
    for (int j = 0; j < dim; j++) {
        temp_array2[j] = best_position[j] + pos[j] * temp_array1[j];
        pos[j] = P * (temp_array2[j] - pos[j]);
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    seeds[gid] = seed; // Update seed
}

// Position adjustment phase
__kernel void position_adjustment_phase(
    __global float *population,
    __global float *best_position,
    __global float *bounds,
    __global uint *seeds,
    float R,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid]; // Copy seed
    float r4 = rand_float(&seed);
    float r4_squared = r4 * r4;
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        pos[j] = pos[j] * R - fabs(best_position[j] - pos[j]) * r4_squared;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    seeds[gid] = seed; // Update seed
}

// Exploitation phase
__kernel void exploitation_phase(
    __global float *population,
    __global float *best_position,
    __global float *second_best,
    __global float *bounds,
    __global uint *seeds,
    float beta,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid]; // Copy seed
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float theta1 = rand_float(&seed);
        float a1 = beta * rand_float(&seed) - beta;
        float b1 = tan(theta1 * M_PI);
        float X1 = best_position[j] - a1 * b1 * fabs(best_position[j] - pos[j]);

        float theta2 = rand_float(&seed);
        float a2 = beta * rand_float(&seed) - beta;
        float b2 = tan(theta2 * M_PI);
        float X2 = second_best[j] - a2 * b2 * fabs(second_best[j] - pos[j]);

        pos[j] = (X1 + X2) / 2.0f;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
    seeds[gid] = seed; // Update seed
}
