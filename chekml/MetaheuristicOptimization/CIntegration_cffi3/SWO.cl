// Optimization parameters
#define SWO_TRADE_OFF 0.3f
#define SWO_CROSSOVER_PROB 0.2f
#define SWO_LEVY_SCALE 0.05f

// Random number generator (XOR-shift)
inline float rand_float(__private uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Generate Levy flight step (optimized for beta = 1.5)
inline float levy_flight_swo(__private uint *seed) {
    const float sigma = 0.696066f; // Precomputed for beta = 1.5
    float u = (rand_float(seed) * 2.0f - 1.0f) * sigma;
    float v = rand_float(seed) * 2.0f - 1.0f;
    if (v == 0.0f) return 0.0f;
    float step = u / pow(fabs(v), 0.666667f); // 1 / 1.5 = 0.666667
    if (step > 1.0f || step < -1.0f || !isfinite(step)) {
        return (step >= 0.0f) ? SWO_LEVY_SCALE : -SWO_LEVY_SCALE;
    }
    return SWO_LEVY_SCALE * step;
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
    uint local_seed = seeds[gid]; // Copy to private memory
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&local_seed) * (upper - lower);
    }
    seeds[gid] = local_seed; // Write back updated seed

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Hunting phase
__kernel void hunting_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global float *best_position,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    uint local_seed = seeds[gid]; // Copy to private memory
    float r1 = rand_float(&local_seed);
    float r2 = rand_float(&local_seed);
    float L = levy_flight_swo(&local_seed);

    float bound_range[32]; // Assuming dim <= 32 for simplicity
    for (int j = 0; j < dim; j++) {
        bound_range[j] = fabs(bounds[j * 2 + 1] - bounds[j * 2]);
    }

    for (int j = 0; j < dim; j++) {
        if (r1 < SWO_TRADE_OFF) {
            pos[j] += L * (best_position[j] - pos[j]);
        } else {
            pos[j] += r2 * bound_range[j] * (rand_float(&local_seed) * 0.2f - 0.1f);
        }
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    seeds[gid] = local_seed; // Write back updated seed
    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Mating phase
__kernel void mating_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    uint local_seed = seeds[gid]; // Copy to private memory
    int mate_idx = (int)(rand_float(&local_seed) * pop_size);
    if (mate_idx == gid || mate_idx >= pop_size) {
        seeds[gid] = local_seed;
        return;
    }

    __global float *mate_pos = population + mate_idx * dim;
    for (int j = 0; j < dim; j++) {
        if (rand_float(&local_seed) < SWO_CROSSOVER_PROB) {
            pos[j] += rand_float(&local_seed) * (mate_pos[j] - pos[j]);
        }
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    seeds[gid] = local_seed; // Write back updated seed
    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}
