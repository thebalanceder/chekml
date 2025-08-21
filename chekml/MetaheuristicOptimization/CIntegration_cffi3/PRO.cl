// Constants
#define SCHEDULE_MIN 0.9f
#define SCHEDULE_MAX 1.0f

// Random number generator
float rand_float(__global uint* seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (float)(*seed) / (float)0x7fffffffu;
}

// Initialize population kernel
__kernel void initialize_population(
    __global float* population,
    __global float* schedules,
    __global float* fitness,
    __global float* bounds,
    __global float* best_position,
    __global uint* seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    for (int j = 0; j < dim; j++) {
        // Initialize population within bounds
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        population[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        schedules[gid * dim + j] = SCHEDULE_MIN + rand_float(&seeds[gid]) * (SCHEDULE_MAX - SCHEDULE_MIN);
    }
    fitness[gid] = INFINITY; // Initial fitness
    if (gid == 0) {
        for (int j = 0; j < dim; j++) {
            best_position[j] = population[j];
        }
    }
}

// Stimulate behaviors kernel
__kernel void stimulate_behaviors(
    __global float* population,
    __global float* schedules,
    __global float* best_position,
    __global float* other_position,
    __global float* new_population,
    __global uint* seeds,
    __global int* selected_behaviors,
    int landa,
    float tau,
    int dim,
    int pop_size,
    __global float* bounds
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    for (int j = 0; j < dim; j++) {
        new_population[gid * dim + j] = population[gid * dim + j];
    }

    for (int j = 0; j < landa; j++) {
        int idx = selected_behaviors[j];
        float r = rand_float(&seeds[gid]);
        float new_pos = population[gid * dim + idx] +
                        schedules[gid * dim + idx] * (best_position[idx] - population[gid * dim + idx]) +
                        schedules[gid * dim + idx] * tau * (other_position[idx] - population[gid * dim + idx]) * r;
        // Clamp to bounds using built-in clamp
        new_population[gid * dim + idx] = clamp(new_pos, bounds[idx * 2], bounds[idx * 2 + 1]);
    }
}

// Reschedule kernel
__kernel void reschedule(
    __global float* population,
    __global float* schedules,
    __global float* fitness,
    __global float* bounds,
    __global uint* seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    for (int j = 0; j < dim; j++) {
        if (rand_float(&seeds[gid]) < 0.1f) {
            schedules[gid * dim + j] = SCHEDULE_MIN + rand_float(&seeds[gid]) * (SCHEDULE_MAX - SCHEDULE_MIN);
        }
    }
}
