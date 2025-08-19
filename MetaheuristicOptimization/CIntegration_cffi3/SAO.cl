#define SAO_BROWNIAN_VARIANCE 0.5f
#define SAO_LOW -5.0f
#define SAO_UP 5.0f
#define SAO_EULER 2.718281828459045f
#define SAO_DF_MIN 0.35f
#define SAO_DF_MAX 0.6f
// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    float r = (*seed) / 4294967296.0f;
    return r < 1e-10f ? 1e-10f : r;
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
        float r = rand_float(&seeds[gid]);
        pos[j] = clamp(SAO_LOW + r * (SAO_UP - SAO_LOW), SAO_LOW, SAO_UP);
    }
    fitness[gid] = INFINITY;
}
// Brownian motion
__kernel void brownian_motion(
    __global float *brownian,
    __global uint *seeds,
    int dim,
    int num_pop
) {
    int gid = get_global_id(0);
    if (gid >= num_pop) return;
    __global float *motion = brownian + gid * dim;
    for (int j = 0; j < dim; j++) {
        float u1 = rand_float(&seeds[gid]);
        float u2 = rand_float(&seeds[gid]);
        float log_u1 = u1 > 1e-10f ? log(u1) : log(1e-10f);
        float value = sqrt(-2.0f * log_u1) * cos(2.0f * M_PI * u2) * SAO_BROWNIAN_VARIANCE;
        motion[j] = isfinite(value) ? clamp(value, -1.0f, 1.0f) : 0.0f;
    }
}
// Calculate centroid
__kernel void calculate_centroid(
    __global float *population,
    __global float *centroid,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= dim) return;
    float sum = 0.0f;
    for (int i = 0; i < pop_size; i++) {
        sum += population[i * dim + gid];
    }
    float result = pop_size > 0 ? sum / pop_size : 0.0f;
    centroid[gid] = isfinite(result) ? result : 0.0f;
}
// Exploration phase
__kernel void exploration_phase(
    __global float *population,
    __global float *brownian,
    __global float *centroid,
    __global float *elite,
    __global float *best_position,
    __global uint *seeds,
    float R,
    int dim,
    int num_a
) {
    int gid = get_global_id(0);
    if (gid >= num_a) return;
    __global float *pos = population + gid * dim;
    float alpha1 = rand_float(&seeds[gid]);
    for (int j = 0; j < dim; j++) {
        float delta_best = best_position[j] - pos[j];
        float delta_centroid = centroid[j] - pos[j];
        float update = R * brownian[gid * dim + j] * (
            alpha1 * delta_best + (1.0f - alpha1) * delta_centroid
        );
        float new_pos = elite[j] + (isfinite(update) ? update : 0.0f);
        pos[j] = clamp(new_pos, SAO_LOW, SAO_UP);
    }
}
// Development phase (simplified covariance matrix learning)
__kernel void development_phase(
    __global float *population,
    __global uint *seeds,
    float R,
    int dim,
    int num_a,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid < num_a || gid >= pop_size) return;
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float mean = 0.0f;
        for (int i = 0; i < pop_size; i++) {
            mean += population[i * dim + j];
        }
        mean = pop_size > 0 ? mean / pop_size : 0.0f;
        float cov_diag = 0.0f;
        for (int i = 0; i < pop_size; i++) {
            float diff = population[i * dim + j] - mean;
            cov_diag += diff * diff;
        }
        float denom = pop_size > 1 ? (pop_size - 1) : 1;
        float cov_result = sqrt(cov_diag / denom + 1e-6f);
        float update = R * pos[j] * (isfinite(cov_result) ? cov_result : 1e-6f);
        pos[j] = clamp(pos[j] + (isfinite(update) ? update : 0.0f), SAO_LOW, SAO_UP);
    }
}
// Random centroid reverse learning
__kernel void random_centroid_reverse_learning(
    __global float *population,
    __global float *reverse_pop,
    __global uint *seeds,
    int dim,
    int pop_size,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;
    // Select B random individuals
    int indices[16];
    for (int i = 0; i < B; i++) {
        float r = rand_float(&seeds[gid]);
        indices[i] = (int)(r * pop_size);
    }
    // Calculate centroid
    float centroid[32];
    for (int j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            sum += population[indices[i] * dim + j];
        }
        centroid[j] = B > 0 ? sum / B : 0.0f;
    }
    // Compute reverse population
    __global float *rpos = reverse_pop + gid * dim;
    for (int j = 0; j < dim; j++) {
        float value = 2.0f * centroid[j] - population[gid * dim + j];
        rpos[j] = clamp(isfinite(value) ? value : 0.0f, SAO_LOW, SAO_UP);
    }
}
