#define MFO_B_CONSTANT 1.0f
#define TWO_PI 6.283185307179586f

// Simple random number generator
uint rng(uint* seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffff;
    return *seed;
}

float random_float(uint* seed, float min, float max) {
    uint r = rng(seed);
    return min + (max - min) * ((float)r / 0x7fffffff);
}

// Initialize population
__kernel void initialize_population(
    __global float* population,
    __global float* fitness,
    __global float* bounds,
    __global uint* seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid];
    for (int j = 0; j < dim; j++) {
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        population[gid * dim + j] = random_float(&seed, lb, ub);
    }
    seeds[gid] = seed;
    fitness[gid] = 0.0f; // Initialize fitness
}

// Update moth positions
__kernel void update_moths(
    __global float* population,
    __global float* fitness,
    __global float* best_flames,
    __global float* bounds,
    __global uint* seeds,
    float a,
    int flame_no,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    uint seed = seeds[gid];
    int flame_idx = min(gid, flame_no - 1);
    float t = random_float(&seed, -1.0f, 1.0f);
    float b = MFO_B_CONSTANT;

    for (int j = 0; j < dim; j++) {
        float moth_pos = population[gid * dim + j];
        float flame_pos = best_flames[flame_idx * dim + j];
        float distance = fabs(flame_pos - moth_pos);
        float new_pos;

        if (gid < flame_no) {
            new_pos = distance * exp(b * t) * cos(TWO_PI * t) + flame_pos;
        } else {
            new_pos = distance * exp(b * t) * cos(TWO_PI * t) + best_flames[0];
        }

        // Bound checking
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        population[gid * dim + j] = clamp(new_pos, lb, ub);
    }

    seeds[gid] = seed;
}

// Bitonic sort (simplified for power of 2 sizes)
__kernel void bitonic_sort(
    __global float* fitness,
    __global int* indices,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size / 2) return;

    for (int stage = 0; stage < (int)log2((float)pop_size); stage++) {
        for (int step = stage; step >= 0; step--) {
            int group = gid >> step;
            int dir = (group & 1) ? 1 : 0;
            int stride = 1 << step;
            int pos = (gid & (stride - 1)) + (group << step) * 2;

            int idx1 = pos;
            int idx2 = pos + stride;
            if (idx2 < pop_size) {
                float f1 = fitness[idx1];
                float f2 = fitness[idx2];
                int i1 = indices[idx1];
                int i2 = indices[idx2];

                if ((f1 > f2 && dir == 0) || (f1 < f2 && dir == 1)) {
                    fitness[idx1] = f2;
                    fitness[idx2] = f1;
                    indices[idx1] = i2;
                    indices[idx2] = i1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

// Update flames on GPU
__kernel void update_flames(
    __global float* population,
    __global float* fitness,
    __global float* best_flames,
    __global float* best_flame_fitness,
    __global float* temp_population,
    __global float* temp_fitness,
    int pop_size,
    int dim
) {
    int gid = get_global_id(0);
    if (gid >= 2 * pop_size) return;

    // Combine populations
    if (gid < pop_size) {
        // Copy current population
        for (int j = 0; j < dim; j++) {
            temp_population[gid * dim + j] = population[gid * dim + j];
        }
        temp_fitness[gid] = fitness[gid];
    } else {
        // Copy best flames
        int idx = gid - pop_size;
        for (int j = 0; j < dim; j++) {
            temp_population[gid * dim + j] = best_flames[idx * dim + j];
        }
        temp_fitness[gid] = best_flame_fitness[idx];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Simplified sort (assumes power of 2)
    for (int stage = 0; stage < (int)log2((float)(2 * pop_size)); stage++) {
        for (int step = stage; step >= 0; step--) {
            int group = gid >> step;
            int dir = (group & 1) ? 1 : 0;
            int stride = 1 << step;
            int pos = (gid & (stride - 1)) + (group << step) * 2;

            if (pos + stride < 2 * pop_size) {
                float f1 = temp_fitness[pos];
                float f2 = temp_fitness[pos + stride];

                if ((f1 > f2 && dir == 0) || (f1 < f2 && dir == 1)) {
                    // Swap fitness
                    temp_fitness[pos] = f2;
                    temp_fitness[pos + stride] = f1;

                    // Swap positions
                    for (int j = 0; j < dim; j++) {
                        float tmp = temp_population[pos * dim + j];
                        temp_population[pos * dim + j] = temp_population[(pos + stride) * dim + j];
                        temp_population[(pos + stride) * dim + j] = tmp;
                    }
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }

    // Update best flames with top pop_size entries
    if (gid < pop_size) {
        for (int j = 0; j < dim; j++) {
            best_flames[gid * dim + j] = temp_population[gid * dim + j];
        }
        best_flame_fitness[gid] = temp_fitness[gid];
    }
}
