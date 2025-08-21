#define PVS_DISTRIBUTION_INDEX 20.0f
#define PVS_X_GAMMA 0.1f
#define M_PI 3.14159265358979323846f

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

// Kernel to initialize vortex
__kernel void init_vortex(__global float *positions,
                          __global float *center,
                          __global float *bounds,
                          __global uint *seeds,
                          int dim,
                          int population_size) {
    int idx = get_global_id(0);
    if (idx >= population_size) return;

    float x = PVS_X_GAMMA;
    float a = 1.0f;
    float ginv = (1.0f / x) * (1.0f / (x * (0.5f + 0.5f * a)));

    for (int j = 0; j < dim; j++) {
        float bound_diff = bounds[2 * j + 1] - bounds[2 * j];
        float radius = ginv * bound_diff / 2.0f;
        float u1 = rand_float(seeds, idx);
        float u2 = rand_float(seeds, idx);
        float z = sqrt(-2.0f * log(max(u1, 1e-10f))) * cos(2.0f * M_PI * u2);
        float pos = center[j] + radius * z;
        positions[idx * dim + j] = clamp(pos, bounds[2 * j], bounds[2 * j + 1]);
    }
}

// Kernel for first phase
__kernel void first_phase(__global float *positions,
                          __global float *center,
                          __global float *bounds,
                          __global uint *seeds,
                          int dim,
                          int size,
                          float radius_factor) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    int use_z2 = 0;
    float z1, z2;
    for (int j = 0; j < dim; j++) {
        float bound_diff = bounds[2 * j + 1] - bounds[2 * j];
        float radius = radius_factor * bound_diff / 2.0f;
        if (!use_z2) {
            float u1 = rand_float(seeds, idx);
            float u2 = rand_float(seeds, idx);
            float r = sqrt(-2.0f * log(max(u1, 1e-10f)));
            z1 = r * cos(2.0f * M_PI * u2);
            z2 = r * sin(2.0f * M_PI * u2);
            positions[idx * dim + j] = center[j] + radius * z1;
            use_z2 = 1;
        } else {
            positions[idx * dim + j] = center[j] + radius * z2;
            use_z2 = 0;
        }
        positions[idx * dim + j] = clamp(positions[idx * dim + j], bounds[2 * j], bounds[2 * j + 1]);
    }
}

// Kernel to compute roulette wheel probabilities
__kernel void compute_probabilities(__global float *costs,
                                   __global float *probabilities,
                                   int population_size) {
    int idx = get_global_id(0);
    if (idx != 0) return; // Only one thread computes probabilities

    float max_val = costs[0];
    for (int k = 1; k < population_size; k++) {
        if (costs[k] > max_val) max_val = costs[k];
    }
    float sum_prob = 0.0f;
    for (int k = 0; k < population_size; k++) {
        probabilities[k] = 0.9f * (max_val - costs[k]) + 0.1f;
        sum_prob += probabilities[k];
    }
    if (sum_prob > 0.0f) {
        probabilities[0] /= sum_prob;
        for (int k = 1; k < population_size; k++) {
            probabilities[k] = probabilities[k - 1] + (probabilities[k] / sum_prob);
        }
    }
}

// Kernel for crossover and mutation
__kernel void crossover_mutation(__global float *positions,
                                __global float *costs,
                                __global float *probabilities,
                                __global float *temp_solution,
                                __global float *mutated_solution,
                                __global float *bounds,
                                __global uint *seeds,
                                int dim,
                                int population_size,
                                float prob_mut,
                                float prob_cross) {
    int idx = get_global_id(0);
    if (idx >= population_size / 2) return;

    int i = idx + population_size / 2;

    // Roulette wheel selection
    float r = rand_float(seeds, i);
    int neighbor = population_size - 1;
    for (int j = 0; j < population_size; j++) {
        if (r <= probabilities[j]) {
            neighbor = j;
            break;
        }
    }
    while (i == neighbor) {
        r = rand_float(seeds, i);
        neighbor = population_size - 1;
        for (int j = 0; j < population_size; j++) {
            if (r <= probabilities[j]) {
                neighbor = j;
                break;
            }
        }
    }

    // Crossover
    int param2change = xorshift32(seeds, i) % dim;
    for (int j = 0; j < dim; j++) {
        temp_solution[i * dim + j] = positions[i * dim + j];
        if (rand_float(seeds, i) < prob_cross || j == param2change) {
            float diff = positions[i * dim + j] - positions[neighbor * dim + j];
            temp_solution[i * dim + j] += diff * (rand_float(seeds, i) - 0.5f) * 2.0f;
            temp_solution[i * dim + j] = clamp(temp_solution[i * dim + j], bounds[2 * j], bounds[2 * j + 1]);
        }
    }

    // Polynomial mutation
    float mut_pow = 1.0f / (1.0f + PVS_DISTRIBUTION_INDEX);
    int state = 0;
    for (int j = 0; j < dim; j++) {
        mutated_solution[i * dim + j] = temp_solution[i * dim + j];
        if (rand_float(seeds, i) <= prob_mut) {
            float y = temp_solution[i * dim + j];
            float yL = bounds[2 * j];
            float yU = bounds[2 * j + 1];
            float delta1 = (y - yL) / (yU - yL);
            float delta2 = (yU - y) / (yU - yL);
            float rnd = rand_float(seeds, i);
            float xy, val, deltaq;

            if (rnd <= 0.5f) {
                xy = 1.0f - delta1;
                val = 2.0f * rnd + (1.0f - 2.0f * rnd) * pow(xy, PVS_DISTRIBUTION_INDEX + 1.0f);
                deltaq = pow(val, mut_pow) - 1.0f;
            } else {
                xy = 1.0f - delta2;
                val = 2.0f * (1.0f - rnd) + 2.0f * (rnd - 0.5f) * pow(xy, PVS_DISTRIBUTION_INDEX + 1.0f);
                deltaq = 1.0f - pow(val, mut_pow);
            }

            y = y + deltaq * (yU - yL);
            mutated_solution[i * dim + j] = clamp(y, yL, yU);
            state++;
        }
    }
}
