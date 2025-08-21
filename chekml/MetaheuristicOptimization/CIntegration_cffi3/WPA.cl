#define WPA_R1_MIN 0.0f
#define WPA_R1_MAX 2.0f
#define WPA_R2_MIN 0.0f
#define WPA_R2_MAX 1.0f
#define WPA_R3_MIN 0.0f
#define WPA_R3_MAX 2.0f
#define WPA_F_MIN -5.0f
#define WPA_F_MAX 5.0f
#define WPA_C_MIN -5.0f
#define WPA_C_MAX 5.0f
#define WPA_STAGNATION_THRESHOLD 3
#define PI 3.14159265358979323846f

// Random number generator (XOR-shift)
inline float rand_float(__private uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Gaussian random number generator (Box-Muller transform)
inline float gaussian(float mu, float sigma, __private uint *seed) {
    float u1 = rand_float(seed);
    float u2 = rand_float(seed);
    if (u1 <= 0.0f) u1 = 1e-10f; // Prevent log(0)
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * PI * u2;
    return mu + sigma * (r * cos(theta));
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
    __private uint seed = seeds[gid]; // Copy global seed to private memory
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seed) * (upper - lower);
    }
    seeds[gid] = seed; // Update global seed
    fitness[gid] = INFINITY;
}

// Compute population mean and standard deviation
__kernel void compute_stats(
    __global float *population,
    __global float *mu_P,
    __global float *sigma,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= dim) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < pop_size; i++) {
        float val = population[i * dim + gid];
        sum += val;
        sum_sq += val * val;
    }
    mu_P[gid] = sum / (float)pop_size;
    float variance = (sum_sq / (float)pop_size) - (mu_P[gid] * mu_P[gid]);
    sigma[gid] = sqrt(variance > 0.0f ? variance : 0.0f);
}

// Exploration Phase
__kernel void wpa_exploration_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *stagnation_counts,
    __global float *mu_P,
    __global float *sigma,
    float K,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __private uint seed = seeds[gid]; // Copy global seed to private memory
    float r1 = WPA_R1_MIN + rand_float(&seed) * (WPA_R1_MAX - WPA_R1_MIN);
    float r2 = WPA_R2_MIN + rand_float(&seed) * (WPA_R2_MAX - WPA_R2_MIN);
    float W[32]; // Max dim = 32
    float new_position[32];

    // Equation (4): W = r1 * (P(t) + 2K)
    for (int j = 0; j < dim; j++) {
        W[j] = r1 * (pos[j] + 2.0f * K);
    }

    // Equation (5): P(t+1) = P(t) + W * (2K + r2)
    float factor = 2.0f * K + r2;
    for (int j = 0; j < dim; j++) {
        new_position[j] = pos[j] + W[j] * factor;
        new_position[j] = clamp(new_position[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Update position
    for (int j = 0; j < dim; j++) {
        pos[j] = new_position[j];
    }
    fitness[gid] = INFINITY; // Fitness evaluated on CPU

    // Handle stagnation
    int stagnation = stagnation_counts[gid];
    if (stagnation >= WPA_STAGNATION_THRESHOLD) {
        // Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
        for (int j = 0; j < dim; j++) {
            float gaussian_term = gaussian(mu_P[j], sigma[j], &seed);
            float term = (W[j] != 0.0f) ? r1 * (pos[j] + 2.0f * K) / W[j] : 0.0f;
            new_position[j] = gaussian_term + term;
            new_position[j] = clamp(new_position[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
        for (int j = 0; j < dim; j++) {
            pos[j] = new_position[j];
        }
        stagnation_counts[gid] = 0;
        fitness[gid] = INFINITY;
    }
    seeds[gid] = seed; // Update global seed
}

// Exploitation Phase
__kernel void wpa_exploitation_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *stagnation_counts,
    __global float *best_position,
    float K,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __private uint seed = seeds[gid]; // Copy global seed to private memory
    float r3 = WPA_R3_MIN + rand_float(&seed) * (WPA_R3_MAX - WPA_R3_MIN);
    float W[32]; // Max dim = 32
    float new_position[32];

    // Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
    for (int j = 0; j < dim; j++) {
        W[j] = r3 * (K * best_position[j] + r3 * pos[j]);
    }

    // Equation (8): P(t+1) = P(t) + K * W
    for (int j = 0; j < dim; j++) {
        new_position[j] = pos[j] + K * W[j];
        new_position[j] = clamp(new_position[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Update position
    for (int j = 0; j < dim; j++) {
        pos[j] = new_position[j];
    }
    fitness[gid] = INFINITY; // Fitness evaluated on CPU

    // Handle stagnation
    int stagnation = stagnation_counts[gid];
    if (stagnation >= WPA_STAGNATION_THRESHOLD) {
        // Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
        float r1 = WPA_R1_MIN + rand_float(&seed) * (WPA_R1_MAX - WPA_R1_MIN);
        float f = WPA_F_MIN + rand_float(&seed) * (WPA_F_MAX - WPA_F_MIN);
        float c = WPA_C_MIN + rand_float(&seed) * (WPA_C_MAX - WPA_C_MIN);
        float theta = rand_float(&seed) * 2.0f * PI;
        float factor = (c != 0.0f) ? (r1 + K) * sin((f / c) * theta) : 0.0f;

        for (int j = 0; j < dim; j++) {
            new_position[j] = factor;
            new_position[j] = clamp(new_position[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
        for (int j = 0; j < dim; j++) {
            pos[j] = new_position[j];
        }
        stagnation_counts[gid] = 0;
        fitness[gid] = INFINITY;
    }
    seeds[gid] = seed; // Update global seed
}
