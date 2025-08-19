#define MFO_B_CONSTANT 1.0f
#define TWO_PI 6.283185307179586f

// Random number generator (XOR-shift)
inline float rand_float(__global uint *seed) {
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

    // Fitness will be evaluated separately
    fitness[gid] = INFINITY;
}

// Objective function definition (to be replaced at runtime)
inline float objective_function(__global float *pos, int dim) {
    float sum = 0.0f;
    for (int j = 0; j < dim; j++) {
        sum += pos[j] * pos[j];
    }
    return sum;
    return 0.0f; // Placeholder return to avoid empty function
}

// Evaluate fitness
__kernel void evaluate_fitness(
    __global float *population,
    __global float *fitness,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    fitness[gid] = objective_function(pos, dim);
}

// Update flames (with parallel sorting approximation)
__kernel void update_flames(
    __global float *population,
    __global float *fitness,
    __global float *best_flames,
    __global float *best_flame_fitness,
    __global float *sorted_population,
    __global float *sorted_fitness,
    __global float *best_position,
    __global float *best_fitness,
    int dim,
    int pop_size,
    int iteration
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Copy to sorted buffers
    __global float *pop = population + gid * dim;
    __global float *sorted_pop = sorted_population + gid * dim;
    for (int j = 0; j < dim; j++) {
        sorted_pop[j] = pop[j];
    }
    sorted_fitness[gid] = fitness[gid];

    // Barrier to ensure all copies are done
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Parallel bubble sort (more passes for better sorting)
    for (int i = 0; i < pop_size; i++) {
        int idx = gid * 2 + (i % 2);
        if (idx + 1 < pop_size) {
            if (sorted_fitness[idx] > sorted_fitness[idx + 1]) {
                // Swap fitness
                float temp_f = sorted_fitness[idx];
                sorted_fitness[idx] = sorted_fitness[idx + 1];
                sorted_fitness[idx + 1] = temp_f;

                // Swap positions
                __global float *pos1 = sorted_population + idx * dim;
                __global float *pos2 = sorted_population + (idx + 1) * dim;
                for (int j = 0; j < dim; j++) {
                    float temp_p = pos1[j];
                    pos1[j] = pos2[j];
                    pos2[j] = temp_p;
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Update flames
    __global float *flame = best_flames + gid * dim;
    if (iteration == 0) {
        // Initialize flames with sorted population
        for (int j = 0; j < dim; j++) {
            flame[j] = sorted_pop[j];
        }
        best_flame_fitness[gid] = sorted_fitness[gid];
    } else {
        // Combine current and previous flames
        if (sorted_fitness[gid] < best_flame_fitness[gid]) {
            for (int j = 0; j < dim; j++) {
                flame[j] = sorted_pop[j];
            }
            best_flame_fitness[gid] = sorted_fitness[gid];
        }
    }

    // Update global best
    if (gid == 0) {
        float min_fitness = sorted_fitness[0];
        int min_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (sorted_fitness[i] < min_fitness) {
                min_fitness = sorted_fitness[i];
                min_idx = i;
            }
        }
        *best_fitness = min_fitness;
        for (int j = 0; j < dim; j++) {
            best_position[j] = sorted_population[min_idx * dim + j];
        }
    }
}

// Update moth positions
__kernel void update_moths(
    __global float *population,
    __global float *fitness,
    __global float *best_flames,
    __global float *best_flame_fitness,
    __global float *sorted_population,
    __global float *sorted_fitness,
    __global float *bounds,
    __global uint *seeds,
    float a,
    int flame_no,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float t_rand = (a - 1.0f) * rand_float(&seeds[gid]) + 1.0f;
    float spiral = exp(MFO_B_CONSTANT * t_rand) * cos(t_rand * TWO_PI);

    if (gid < flame_no) {
        // Update w.r.t. corresponding flame
        __global float *flame = sorted_population + gid * dim;
        for (int j = 0; j < dim; j++) {
            float distance_to_flame = fabs(flame[j] - pos[j]);
            pos[j] = distance_to_flame * spiral + flame[j];
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        // Update w.r.t. best flame
        __global float *best_flame = sorted_population + (flame_no - 1) * dim;
        for (int j = 0; j < dim; j++) {
            float distance_to_flame = fabs(best_flame[j] - pos[j]);
            pos[j] = distance_to_flame * spiral + best_flame[j];
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Fitness will be evaluated separately
    fitness[gid] = INFINITY;
}
