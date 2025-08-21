// OpenCL kernel source for Harris Hawks Optimization

// Constants
#define HHO_BETA 1.5f
#define PI 3.14159265358979323846f
#define LEVY_SIGMA 0.696568676784f  // Precomputed for beta=1.5

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
    __global float *best_position,
    __global float *best_fitness,
    __global float *bounds,
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

    // Placeholder fitness (updated on CPU)
    fitness[gid] = 0.0f;

    // Initialize best (simplified, actual update on CPU)
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (gid == 0) {
        *best_fitness = INFINITY;
        for (int j = 0; j < dim; j++) best_position[j] = pos[j];
    }
}

// Levy flight using Box-Muller transform
__kernel void levy_flight(
    __global float *step,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *step_vec = step + gid * dim;
    for (int i = 0; i < dim; i += 2) {
        float r1 = rand_float(&seeds[gid]);
        float r2 = rand_float(&seeds[gid]);
        float z0 = sqrt(-2.0f * log(r1)) * cos(2.0f * PI * r2);
        float z1 = sqrt(-2.0f * log(r1)) * sin(2.0f * PI * r2);
        step_vec[i] = LEVY_SIGMA * z0 / pow(fabs(rand_float(&seeds[gid])), 1.0f / HHO_BETA);
        if (i + 1 < dim) {
            step_vec[i + 1] = LEVY_SIGMA * z1 / pow(fabs(rand_float(&seeds[gid])), 1.0f / HHO_BETA);
        }
    }
}

// Exploration phase
__kernel void exploration_phase(
    __global float *population,
    __global float *fitness,
    __global float *best_position,
    __global float *mean_pos,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float q = rand_float(&seeds[gid]);
    int rand_hawk_idx = (int)(rand_float(&seeds[gid]) * pop_size);
    if (rand_hawk_idx == pop_size) rand_hawk_idx--;

    if (q < 0.5f) {
        // Perch based on other family members
        for (int j = 0; j < dim; j++) {
            float r1 = rand_float(&seeds[gid]);
            float r2 = rand_float(&seeds[gid]);
            pos[j] = population[rand_hawk_idx * dim + j] - 
                     r1 * fabs(population[rand_hawk_idx * dim + j] - 2.0f * r2 * pos[j]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        // Perch on a random tall tree
        for (int j = 0; j < dim; j++) {
            float r = rand_float(&seeds[gid]);
            pos[j] = (best_position[j] - mean_pos[j]) - 
                     r * ((bounds[j * 2 + 1] - bounds[j * 2]) * rand_float(&seeds[gid]) + bounds[j * 2]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }
}

// Exploitation phase
__kernel void exploitation_phase(
    __global float *population,
    __global float *fitness,
    __global float *best_position,
    float best_fitness,
    __global float *mean_pos,
    __global float *levy_step,
    float escaping_energy,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *levy = levy_step + gid * dim;
    float r = rand_float(&seeds[gid]);
    float jump_strength = 2.0f * (1.0f - rand_float(&seeds[gid]));
    float curr_fitness = fitness[gid];

    if (r >= 0.5f && fabs(escaping_energy) < 0.5f) {
        // Hard besiege
        for (int j = 0; j < dim; j++) {
            pos[j] = best_position[j] - 
                     escaping_energy * fabs(best_position[j] - pos[j]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else if (r >= 0.5f && fabs(escaping_energy) >= 0.5f) {
        // Soft besiege
        for (int j = 0; j < dim; j++) {
            pos[j] = (best_position[j] - pos[j]) - 
                     escaping_energy * fabs(jump_strength * best_position[j] - pos[j]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else if (r < 0.5f && fabs(escaping_energy) >= 0.5f) {
        // Soft besiege with rapid dives
        float X1[32]; // Assuming dim <= 32 for simplicity
        for (int j = 0; j < dim; j++) {
            X1[j] = best_position[j] - 
                    escaping_energy * fabs(jump_strength * best_position[j] - pos[j]);
            X1[j] = clamp(X1[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
        // Fitness evaluated on CPU, assume X1 is better for now
        for (int j = 0; j < dim; j++) {
            pos[j] = X1[j];
        }
        // Note: Levy flight step is precomputed, no need for X2 here
    } else {
        // Hard besiege with rapid dives
        float X1[32];
        for (int j = 0; j < dim; j++) {
            X1[j] = best_position[j] - 
                    escaping_energy * fabs(jump_strength * best_position[j] - mean_pos[j]);
            X1[j] = clamp(X1[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
        // Fitness evaluated on CPU, assume X1 is better for now
        for (int j = 0; j < dim; j++) {
            pos[j] = X1[j];
        }
    }
}
