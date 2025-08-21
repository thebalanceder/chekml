#define EVO_STEP_SIZE 0.1f
#define EVO_MOMENTUM 0.9f
#define EVO_LEARNING_RATE 0.2f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize particles
__kernel void initialize_particles(
    __global float *position,
    __global float *velocity,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Initialize position and velocity
    __global float *pos = position + gid * dim;
    __global float *vel = velocity + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        vel[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Update velocities and positions
__kernel void update_velocity_and_position(
    __global float *position,
    __global float *velocity,
    __global float *gradient,
    __global float *bounds,
    float step_size,
    float momentum,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = position + gid * dim;
    __global float *vel = velocity + gid * dim;
    __global float *grad = gradient + gid * dim;

    for (int j = 0; j < dim; j++) {
        vel[j] = momentum * vel[j] + step_size * grad[j];
        pos[j] -= vel[j];
        // Enforce bounds
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }
}
