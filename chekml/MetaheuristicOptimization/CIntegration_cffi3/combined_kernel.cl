// Linear congruential generator
uint lcg_rand(uint* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed >> 16) & 0x7FFF;
}
float lcg_rand_float(uint* seed) {
    return ((float)lcg_rand(seed) / 0x7FFF);
}
__kernel void initialize_population(
    __global float* population,
    __global const float* bounds,
    const int dim,
    const int population_size,
    uint seed)
{
    int id = get_global_id(0);
    if (id < population_size) {
        uint local_seed = seed + id * dim;
        for (int d = 0; d < dim; d++) {
            float min = bounds[2 * d];
            float max = bounds[2 * d + 1];
            population[id * dim + d] = min + (max - min) * lcg_rand_float(&local_seed);
        }
    }
}
#define HRO 1.2f
#define HRI 7.2f
#define HGO 1.3f
#define HGI 0.82f
#define CFR_FACTOR 9.435f
__kernel void diversion_phase(
    __global float* population,
    __global const float* best,
    __global const float* bounds,
    const int dim,
    const int population_size,
    uint seed,
    __local float* local_best,
    __local float* local_bounds)
{
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    if (id < population_size) {
        uint local_seed = seed + id * dim;
        if (local_id < dim) {
            local_best[local_id] = best[local_id];
            local_bounds[2 * local_id] = bounds[2 * local_id];
            local_bounds[2 * local_id + 1] = bounds[2 * local_id + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float r1 = lcg_rand_float(&local_seed);
        float r2 = lcg_rand_float(&local_seed);
        float CFR = CFR_FACTOR * lcg_rand_float(&local_seed) * 2.5f;
        float velocity = (r1 < 0.23f) ? (pow(HRO, 2.0f / 3.0f) * sqrt(HGO) / CFR) * r1
                                      : (pow(HRI, 2.0f / 3.0f) * sqrt(HGI) / CFR) * r2;
        for (int d = 0; d < dim; d++) {
            float delta = velocity * (local_best[d] - population[id * dim + d]) * lcg_rand_float(&local_seed);
            float new_pos = population[id * dim + d] + delta;
            population[id * dim + d] = clamp(new_pos, local_bounds[2 * d], local_bounds[2 * d + 1]);
        }
    }
}
#define WATER_DENSITY 1.35f
#define FLUID_DISTRIBUTION 0.46f
#define CENTRIFUGAL_RESISTANCE 1.2f
__kernel void spiral_motion_update(
    __global float* population,
    __global const float* fitness,
    __global const float* best,
    const float best_fitness,
    __global const float* bounds,
    const int dim,
    const int population_size,
    const float t,
    const float max_iter,
    uint seed,
    __local float* local_best,
    __local float* local_bounds,
    __local float* local_sums)
{
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    if (id < population_size) {
        uint local_seed = seed + id * dim;
        if (local_id < dim) {
            local_best[local_id] = best[local_id];
            local_bounds[2 * local_id] = bounds[2 * local_id];
            local_bounds[2 * local_id + 1] = bounds[2 * local_id + 1];
        }
        local_sums[local_id] = id < population_size ? fitness[id] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int offset = local_size / 2; offset > 0; offset /= 2) {
            if (local_id < offset) {
                local_sums[local_id] += local_sums[local_id + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float MLV = local_sums[0] / population_size;
        float LP = (WATER_DENSITY * FLUID_DISTRIBUTION * MLV * MLV) / CENTRIFUGAL_RESISTANCE;
        float RCF = WATER_DENSITY * cos(1.57079632679f * (t / max_iter)) * sqrt(fabs(best_fitness - fitness[id]));
        if (RCF > LP) {
            for (int d = 0; d < dim; d++) {
                float min = local_bounds[2 * d];
                float max = local_bounds[2 * d + 1];
                population[id * dim + d] = min + (max - min) * lcg_rand_float(&local_seed);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int d = 0; d < dim; d++) {
            population[id * dim + d] = clamp(population[id * dim + d], local_bounds[2 * d], local_bounds[2 * d + 1]);
        }
    }
}
#define BOTTLENECK_RATIO 0.68f
__kernel void local_development_phase(
    __global float* population,
    __global const float* best,
    __global const float* bounds,
    const int dim,
    const int population_size,
    uint seed,
    __local float* local_best,
    __local float* local_bounds)
{
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    if (id < population_size) {
        uint local_seed = seed + id * dim;
        if (local_id < dim) {
            local_best[local_id] = best[local_id];
            local_bounds[2 * local_id] = bounds[2 * local_id];
            local_bounds[2 * local_id + 1] = bounds[2 * local_id + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float r3 = lcg_rand_float(&local_seed);
        float CFR = CFR_FACTOR * lcg_rand_float(&local_seed) * 2.5f;
        float velocity = (pow(HRI, 2.0f / 3.0f) * sqrt(HGI) / (2.0f * CFR)) * ((r3 < BOTTLENECK_RATIO) ? r3 : lcg_rand_float(&local_seed));
        for (int d = 0; d < dim; d++) {
            float delta = velocity * (local_best[d] - population[id * dim + d]) * lcg_rand_float(&local_seed);
            float new_pos = population[id * dim + d] + delta;
            population[id * dim + d] = clamp(new_pos, local_bounds[2 * d], local_bounds[2 * d + 1]);
        }
    }
}
#define ELIMINATION_RATIO 0.23f
__kernel void elimination_phase(
    __global float* population,
    __global float* fitness,
    __global const float* bounds,
    const int dim,
    const int population_size,
    uint seed,
    __local float* local_bounds)
{
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    int worst_count = (int)(ELIMINATION_RATIO * population_size);
    if (id < worst_count) {
        uint local_seed = seed + id * dim;
        if (local_id < dim) {
            local_bounds[2 * local_id] = bounds[2 * local_id];
            local_bounds[2 * local_id + 1] = bounds[2 * local_id + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int idx = population_size - id - 1;
        for (int d = 0; d < dim; d++) {
            float min = local_bounds[2 * d];
            float max = local_bounds[2 * d + 1];
            population[idx * dim + d] = min + (max - min) * lcg_rand_float(&local_seed);
        }
        fitness[idx] = INFINITY;
    }
}
__kernel void find_best(
    __global const float* fitness,
    __global float* temp_fitness,
    __global int* temp_indices,
    const int population_size,
    __local float* local_fitness,
    __local int* local_indices)
{
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    if (id < population_size) {
        local_fitness[local_id] = fitness[id];
        local_indices[local_id] = id;
    } else {
        local_fitness[local_id] = INFINITY;
        local_indices[local_id] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset && local_indices[local_id + offset] != -1) {
            if (local_fitness[local_id + offset] < local_fitness[local_id]) {
                local_fitness[local_id] = local_fitness[local_id + offset];
                local_indices[local_id] = local_indices[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        int group_id = get_group_id(0);
        temp_fitness[group_id] = local_fitness[0];
        temp_indices[group_id] = local_indices[0];
    }
}
__kernel void finalize_best(
    __global const float* temp_fitness,
    __global float* best,
    __global float* best_fitness,
    __global const float* population,
    const int dim,
    const int group_count,
    __global const int* temp_indices,
    __local float* local_fitness,
    __local int* local_indices)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    if (local_id < group_count) {
        local_fitness[local_id] = temp_fitness[local_id];
        local_indices[local_id] = temp_indices[local_id];
    } else {
        local_fitness[local_id] = INFINITY;
        local_indices[local_id] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset && local_indices[local_id + offset] != -1) {
            if (local_fitness[local_id + offset] < local_fitness[local_id]) {
                local_fitness[local_id] = local_fitness[local_id + offset];
                local_indices[local_id] = local_indices[local_id + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        int idx = local_indices[0];
        if (idx >= 0 && local_fitness[0] < *best_fitness) {
            *best_fitness = local_fitness[0];
            for (int d = 0; d < dim; d++) {
                best[d] = population[idx * dim + d];
            }
        }
    }
}

__kernel void evaluate_fitness(
    __global const float* population,
    __global float* fitness,
    const int dim,
    const int population_size)
{
    int id = get_global_id(0);
    if (id < population_size) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float x = population[id * dim + d];
            sum += x * x; // Sphere function as fallback
        }
        fitness[id] = sum;
    }
}
